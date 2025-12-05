# src/eval/offline_eval.py
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from src.config import (
    INTERACTIONS_V1,
    LISTINGS_V1,
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
)
from src.eval.metrics import ndcg_at_k, recall_at_k, coverage
from src.models.rerank import ContextualReranker

from src.config import (
    INTERACTIONS_V1,
    LISTINGS_V1,
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    AGG_LOGS_PATH,
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "popularity",
            "retrieval_text",
            "retrieval_multimodal",
            "retrieval_multimodal_rerank",
        ],
    )
    parser.add_argument("--k_eval", type=int, default=10, help="k pour NDCG@k")
    parser.add_argument(
        "--topk_candidates",
        type=int,
        default=200,
        help="Nombre de candidats récupérés avant re-ranking.",
    )
    parser.add_argument(
        "--use_logs",
        type=int,
        default=0,
        help=(
            "1 = utiliser interactions_agg.parquet (logs UI) "
            "au lieu de interactions_v1.parquet"
        ),
    )
    return parser.parse_args()



def load_embeddings() -> Dict[str, np.ndarray]:
    emb_text = np.load(DATA_PROCESSED_DIR / "emb_text_clip.npy")
    emb_fused = np.load(DATA_PROCESSED_DIR / "emb_item_fused.npy")
    item_ids = np.load(DATA_PROCESSED_DIR / "item_ids.npy")
    return {"item_ids": item_ids, "text": emb_text, "fused": emb_fused}


def build_leave_one_out(interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    interactions = interactions.sort_values("timestamp")
    last = interactions.groupby("user_id").tail(1)
    train = interactions.drop(last.index)
    return train, last


def build_popularity_recommender(train: pd.DataFrame):
    item_counts = train["item_id"].value_counts()

    def recommender_fn(user_id: int, k: int):
        return item_counts.index[:k].tolist()

    return recommender_fn, list(item_counts.index)


def build_user_history(train: pd.DataFrame) -> Dict[int, List[int]]:
    return train.groupby("user_id")["item_id"].apply(list).to_dict()


def build_faiss_index(emb: np.ndarray) -> faiss.Index:
    emb = emb.astype("float32")
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index


def retrieval_mode_eval(
    mode: str,
    item_ids: np.ndarray,
    emb_text: np.ndarray,
    emb_fused: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_eval: int,
    topk_candidates: int,
) -> Dict[str, float]:
    """
    mode: 'retrieval_text' | 'retrieval_multimodal' | 'retrieval_multimodal_rerank'
    """
    user_history = build_user_history(train)
    item_id_to_idx = {int(i): idx for idx, i in enumerate(item_ids)}

    # choisir les embeddings pour l'index
    if mode == "retrieval_text":
        emb_items = emb_text.copy()
    else:
        emb_items = emb_fused.copy()

    index = build_faiss_index(emb_items)

    reranker = None
    if mode == "retrieval_multimodal_rerank":
        reranker = ContextualReranker()

    # ground truth
    users = test["user_id"].unique().tolist()
    user_truth = defaultdict(set)
    for _, row in test.iterrows():
        user_truth[row["user_id"]].add(row["item_id"])

    all_items = sorted(train["item_id"].unique())
    all_recs = {}
    ndcgs = []
    recalls = []

    for uid in users:
        hist = user_history.get(uid, [])
        if len(hist) == 0:
            # aucun historique → on skip ce user
            continue
        # embedding user = moyenne des embeddings items vus
        idxs = [item_id_to_idx[i] for i in hist if i in item_id_to_idx]
        if not idxs:
            continue
        if mode == "retrieval_text":
            emb_u_items = emb_text[idxs]
        else:
            emb_u_items = emb_fused[idxs]
        v_user = emb_u_items.mean(axis=0, keepdims=True).astype("float32")
        faiss.normalize_L2(v_user)

        scores, idx = index.search(v_user, topk_candidates)
        idx = idx[0]
        scores = scores[0]
        cand_ids = item_ids[idx]

        if reranker is not None:
            new_ids, new_scores = reranker.rerank(
                candidate_ids=cand_ids,
                base_scores=scores,
                budget_min=None,
                budget_max=None,
                n_guests=None,
                k=k_eval,  # <-- ici on passe k, plus k_final
            )
            recs = new_ids
        else:
            recs = cand_ids[:k_eval].tolist()



        all_recs[uid] = recs
        gt = user_truth[uid]
        ndcgs.append(ndcg_at_k(recs, gt, k=k_eval))
        recalls.append(recall_at_k(recs, gt, k=max(k_eval, 20)))

    metrics = {
        "ndcg@{}".format(k_eval): float(np.mean(ndcgs) if ndcgs else 0.0),
        "recall@{}".format(max(k_eval, 20)): float(np.mean(recalls) if recalls else 0.0),
        "coverage": coverage(all_recs, all_items),
    }
    return metrics

def main(args):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Choix de la source d'interactions :
    # - INTERACTIONS_V1 = reviews Airbnb (anonymisés)
    # - AGG_LOGS_PATH = logs agrégés de TON interface (users réels)
    if getattr(args, "use_logs", 0):
        interactions_path = AGG_LOGS_PATH
        print(f"=== Using LOGS interactions from {interactions_path} ===")
    else:
        interactions_path = INTERACTIONS_V1
        print(f"=== Using REVIEWS interactions from {interactions_path} ===")

    interactions = pd.read_parquet(interactions_path)
    listings = pd.read_parquet(LISTINGS_V1)

    train, test = build_leave_one_out(interactions)


    emb_data = load_embeddings()
    item_ids = emb_data["item_ids"]
    emb_text = emb_data["text"]
    emb_fused = emb_data["fused"]

    modes = (
        ["popularity", "retrieval_text", "retrieval_multimodal", "retrieval_multimodal_rerank"]
        if args.mode == "all"
        else [args.mode]
    )

    results_all = {}

    for mode in modes:
        print(f"Evaluating mode: {mode}")
        if mode == "popularity":
            recommender_fn, all_items = build_popularity_recommender(train)

            users = test["user_id"].unique().tolist()
            user_truth = defaultdict(set)
            for _, row in test.iterrows():
                user_truth[row["user_id"]].add(row["item_id"])

            all_recs = {}
            ndcgs = []
            recalls = []
            for uid in users:
                recs = recommender_fn(uid, args.k_eval)
                all_recs[uid] = recs
                gt = user_truth[uid]
                ndcgs.append(ndcg_at_k(recs, gt, k=args.k_eval))
                recalls.append(recall_at_k(recs, gt, k=max(args.k_eval, 20)))

            metrics = {
                "ndcg@{}".format(args.k_eval): float(np.mean(ndcgs) if ndcgs else 0.0),
                "recall@{}".format(max(args.k_eval, 20)): float(np.mean(recalls) if recalls else 0.0),
                "coverage": coverage(all_recs, all_items),
            }
        else:
            metrics = retrieval_mode_eval(
                mode=mode,
                item_ids=item_ids,
                emb_text=emb_text,
                emb_fused=emb_fused,
                train=train,
                test=test,
                k_eval=args.k_eval,
                topk_candidates=args.topk_candidates,
            )

        results_all[mode] = metrics
        out_file = EXPERIMENTS_DIR / f"offline_eval_{mode}.json"
        with out_file.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics for {mode} to {out_file}")
        print(metrics)

    if len(results_all) > 1:
        out_file = EXPERIMENTS_DIR / "offline_eval_all_modes.json"
        with out_file.open("w") as f:
            json.dump(results_all, f, indent=2)
        print("Saved all modes metrics to", out_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
