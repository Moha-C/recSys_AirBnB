# src/models/retrieval.py
from pathlib import Path
from typing import Tuple, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch

from src.config import DATA_PROCESSED_DIR, LISTINGS_V1
from src.models.precompute_embeddings import load_clip  # pour encoder la requête texte


INDEX_FILE = DATA_PROCESSED_DIR / "faiss_index_fused.bin"

def load_item_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    ids = np.load(DATA_PROCESSED_DIR / "item_ids.npy")
    emb = np.load(DATA_PROCESSED_DIR / "emb_item_fused.npy")
    return ids, emb


def build_faiss_index() -> None:
    ids, emb = load_item_embeddings()
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    index.add(emb.astype("float32"))
    faiss.write_index(index, str(INDEX_FILE))
    print("FAISS index saved to", INDEX_FILE)


def load_faiss_index() -> faiss.Index:
    index = faiss.read_index(str(INDEX_FILE))
    return index


def search_candidates(
    query_emb: np.ndarray,
    top_k: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_emb: [dim] numpy (déjà normalisé)
    """
    ids, emb = load_item_embeddings()
    index = load_faiss_index()
    q = query_emb.astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, idx = index.search(q, top_k)
    idx = idx[0]
    scores = scores[0]
    return ids[idx], scores

# -------------------------------------------------------------------
# CLIP text encoder (lazy load)
# -------------------------------------------------------------------

_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = "cpu"


def _get_clip_text_encoder():
    """
    Charge le modèle CLIP une seule fois (lazy) pour encoder les requêtes texte.
    """
    global _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_DEVICE
    if _CLIP_MODEL is None or _CLIP_TOKENIZER is None:
        model, preprocess, tokenizer, device = load_clip()
        _CLIP_MODEL = model
        _CLIP_TOKENIZER = tokenizer
        _CLIP_DEVICE = device
    return _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_DEVICE


def encode_query_text(query: str) -> np.ndarray:
    """
    Encode une requête texte en embedding CLIP normalisé [dim].
    """
    model, tokenizer, device = _get_clip_text_encoder()
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
    return emb[0].cpu().numpy()


# -------------------------------------------------------------------
# Chargement listings (une seule fois)
# -------------------------------------------------------------------

if LISTINGS_V1.exists():
    _LISTINGS_DF = pd.read_parquet(LISTINGS_V1)
else:
    _LISTINGS_DF = pd.DataFrame()


# -------------------------------------------------------------------
# search_items utilisé par l'API FastAPI
# -------------------------------------------------------------------

def search_items(
    query: str,
    k: int = 50,
    city: Optional[str] = None,
    n_guests: Optional[int] = None,
    budget_min: Optional[float] = None,
    budget_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Recherche des candidats via FAISS (embeddings multimodaux) à partir
    de la requête texte utilisateur.

    - query: description du trip (texte libre)
    - k: nombre de résultats à retourner (l'API passe souvent k*5 pour
      laisser du marge au rerank)
    - city: si précisé, filtre sur la ville (Paris/Lyon/Bordeaux)

    On ne fait PAS ici le filtrage budget/capacité (géré ensuite par rerank_items),
    on fait juste:
        1) encode query
        2) FAISS search
        3) jointure avec listings_v1 + tri par score
    """
    if _LISTINGS_DF.empty or not len(_LISTINGS_DF):
        return pd.DataFrame()

    # 1) encode la requête
    q_emb = encode_query_text(query)  # [dim]

    # 2) FAISS search sur les embeddings items
    item_ids, scores = search_candidates(q_emb, top_k=max(k, 200))

    score_map = {int(i): float(s) for i, s in zip(item_ids, scores)}

    # 3) Jointure avec listings
    df = _LISTINGS_DF[_LISTINGS_DF["id"].isin(score_map.keys())].copy()
    if df.empty:
        return df

    # Ajoute la colonne score (retrieval)
    df["score"] = df["id"].astype(int).map(score_map)

    # 4) Filtre ville si demandé
    if city is not None:
        df = df[df["city"] == city]

    # Tri par score décroissant et limite
    df = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    return df
