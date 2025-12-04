# src/training/train_multimodal.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    LISTINGS_V1,
    INTERACTIONS_V1,
    DATA_PROCESSED_DIR,
    FUSED_EMB_NPY,
    EXPERIMENTS_DIR,
    MODEL_WEIGHTS,
    LOSS_CURVES_PNG,
    METRICS_LAST,
)
from src.models.core_model import MultimodalRecModel, ModelConfig
from src.training.dataset import PairwiseInteractionDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--use_logs",
        type=int,
        default=1,
        help="1 = utiliser interactions_agg.parquet si dispo, sinon interactions_v1",
    )
    parser.add_argument(
        "--alpha_text",
        type=float,
        default=0.7,
        help="(pour info seulement ici, le CLIP est déjà fusionné dans emb_item_fused)",
    )
    return parser.parse_args()


def load_interactions(use_logs: int) -> pd.DataFrame:
    """
    Charge les interactions pour l'entraînement :
    - si use_logs=1 et interactions_agg.parquet existe -> on l'utilise
    - sinon -> fallback sur interactions_v1.parquet
    """
    logs_path = DATA_PROCESSED_DIR / "interactions_agg.parquet"
    if use_logs and logs_path.exists():
        print(f"=== Loading aggregated logs from {logs_path} ===")
        df = pd.read_parquet(logs_path)
    else:
        if use_logs:
            print(
                "WARNING: use_logs=1 mais interactions_agg.parquet introuvable.\n"
                f"Loading default interactions_v1: {INTERACTIONS_V1}"
            )
        else:
            print(f"=== Loading interactions_v1 from {INTERACTIONS_V1} ===")
        df = pd.read_parquet(INTERACTIONS_V1)

    # On s'assure d'avoir user_id / item_id et on supprime les NaN
    needed_cols = {"user_id", "item_id"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in interactions: {missing}")

    df = df.dropna(subset=["user_id", "item_id"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    print(f"Total interactions loaded: {len(df)}")
    return df


def train_val_split_leave_one_out(
    interactions: pd.DataFrame,
    min_items_per_user: int = 2,
    val_items_per_user: int = 1,
):
    """
    Split train/val par utilisateur (simple leave-one-out) :

    - Pour chaque user avec au moins min_items_per_user interactions :
      - on échantillonne val_items_per_user interactions en validation
      - le reste va dans le train
    - Si jamais ça produit un train vide (cas pathologique) -> fallback split global 80/20.
    """
    interactions = interactions.copy()

    # Shuffle pour casser tout biais d'ordre
    interactions = interactions.sample(frac=1.0, random_state=42).reset_index(drop=True)

    groups = interactions.groupby("user_id")

    train_parts = []
    val_parts = []

    rng = np.random.default_rng(42)

    for uid, g in groups:
        if len(g) < min_items_per_user:
            # on ignore ces users pour l'entraînement (trop peu d'infos)
            continue

        n_val = min(val_items_per_user, len(g) - 1)
        idx = g.index.to_numpy()
        val_idx = rng.choice(idx, size=n_val, replace=False)
        mask = np.isin(idx, val_idx, invert=True)

        val_parts.append(g.loc[val_idx])
        train_parts.append(g.loc[idx[mask]])

    if not train_parts:
        print(
            "WARNING: leave-one-out split produced empty train set. "
            "Falling back to a simple global 80/20 split."
        )
        df = interactions
        df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
        split = int(0.8 * len(df))
        train_df = df.iloc[:split].reset_index(drop=True)
        val_df = df.iloc[split:].reset_index(drop=True)
    else:
        train_df = pd.concat(train_parts).reset_index(drop=True)
        val_df = pd.concat(val_parts).reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    return train_df, val_df


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading listings_v1 ===")
    listings = pd.read_parquet(LISTINGS_V1)
    if "id" not in listings.columns:
        raise ValueError("Column 'id' not found in listings_v1.parquet")

    print("=== Loading fused CLIP embeddings ===")
    emb_np = np.load(FUSED_EMB_NPY)  # shape [N_items, dim]
    item_emb = torch.from_numpy(emb_np).float().to(device)

    # On suppose que l'ordre des lignes dans emb_item_fused.npy
    # correspond à l'ordre de listings_v1
    item_ids = listings["id"].astype(int).tolist()
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    print("=== Loading interactions (reviews + logs) ===")
    interactions = load_interactions(args.use_logs)

    # On ne garde que les interactions dont l'item a un embedding
    interactions = interactions[interactions["item_id"].isin(item_id_to_idx.keys())]
    print(f"Interactions after filtering to items with embeddings: {len(interactions)}")

    if len(interactions) == 0:
        raise ValueError("No interactions left after filtering on item_id -> embeddings.")

    # Split train / val robuste
    train_df, val_df = train_val_split_leave_one_out(interactions)

    if len(train_df) == 0:
        raise ValueError("Train set is empty after splitting. Aborting training.")

    # Dataset + DataLoader
    train_dataset = PairwiseInteractionDataset(train_df, item_id_to_idx,user_min_interactions=1,)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Modèle
    config = ModelConfig(embedding_dim=item_emb.shape[1], alpha_text=args.alpha_text)
    model = MultimodalRecModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []

    print("=== Starting training ===")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            user_ids, pos_ids, neg_ids = batch
            user_ids = user_ids.tolist()
            pos_ids = pos_ids.tolist()
            neg_ids = neg_ids.tolist()

            # Map item_id -> index dans emb_item_fused
            pos_idx = [item_id_to_idx[int(i)] for i in pos_ids]
            neg_idx = [item_id_to_idx[int(i)] for i in neg_ids]

            pos_emb = item_emb[pos_idx]  # [B, dim]
            neg_emb = item_emb[neg_idx]  # [B, dim]

            # user embedding = moyenne des items vus (dans le dataset)
            user_emb_list = []
            for uid in user_ids:
                items_u = train_dataset.user_items.get(int(uid), [])
                if not items_u:
                    # fallback: vecteur nul
                    user_emb_list.append(torch.zeros(1, item_emb.shape[1], device=device))
                    continue
                idxs_u = [item_id_to_idx[int(i)] for i in items_u if int(i) in item_id_to_idx]
                if not idxs_u:
                    user_emb_list.append(torch.zeros(1, item_emb.shape[1], device=device))
                    continue
                emb_u = item_emb[idxs_u]
                u = model.encode_user(emb_u)  # [1, dim]
                user_emb_list.append(u)

            user_emb_batch = torch.cat(user_emb_list, dim=0).to(device)

            optimizer.zero_grad()
            loss = model(user_emb_batch, model.encode_item(pos_emb), model.encode_item(neg_emb))
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: train loss = {avg_loss:.4f}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), MODEL_WEIGHTS)
    print("Model saved to", MODEL_WEIGHTS)

    # Courbe de loss
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        LOSS_CURVES_PNG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(LOSS_CURVES_PNG)
        print("Loss curves saved to", LOSS_CURVES_PNG)
    except Exception as e:
        print("Could not save loss curves:", e)

    # Sauvegarde d'un petit JSON de métriques
    metrics = {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "num_epochs": args.epochs,
        "num_train_samples": int(len(train_df)),
        "num_val_samples": int(len(val_df)),
    }
    METRICS_LAST.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LAST.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Final metrics saved to", METRICS_LAST)


if __name__ == "__main__":
    args = parse_args()
    main(args)
