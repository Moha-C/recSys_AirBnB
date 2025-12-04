# src/models/rerank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    LISTINGS_V1,
)

# -------------------------------------------------------------------
# Helpers & constants
# -------------------------------------------------------------------

CONTEXT_FILE = DATA_PROCESSED_DIR / "item_context_features.parquet"

# On garde ça pour compatibilité si jamais tu veux utiliser ce fichier,
# mais le rerank ci-dessous se base surtout sur LISTINGS_V1.
if CONTEXT_FILE.exists():
    _CONTEXT_DF = pd.read_parquet(CONTEXT_FILE)
else:
    _CONTEXT_DF = None


def _safe_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


# -------------------------------------------------------------------
# Rerank "fonctionnel" pour l'API / React
# -------------------------------------------------------------------

def rerank_items(
    candidates: pd.DataFrame,
    user_id: Optional[str] = None,
    budget_min: Optional[float] = None,
    budget_max: Optional[float] = None,
    n_guests: Optional[int] = None,
    k: int = 10,
) -> pd.DataFrame:
    """
    Rerank les candidats en combinant :
    - score de base (venant de la recherche sémantique FAISS)
    - respect de la capacité (accommodates)
    - adéquation au budget (price ~ centre de l'intervalle)
    Retourne un DataFrame trié par score final décroissant, limité à k.
    """
    if candidates.empty:
        return candidates

    df = candidates.copy()

    # Score de base (retrieval)
    if "score" in df.columns:
        base_score = _safe_float(df["score"])
    else:
        base_score = pd.Series(0.0, index=df.index)

    # Prix
    price = _safe_float(df.get("price", pd.Series(0.0, index=df.index)))

    # Capacité
    accommodates = _safe_float(df.get("accommodates", pd.Series(1.0, index=df.index)), default=1.0)

    # ------------------------------------------------------------------
    # 1) Filtre dur sur capacité + budget si précisé
    # ------------------------------------------------------------------
    if n_guests is not None:
        df = df[accommodates >= n_guests]
        base_score = base_score.loc[df.index]
        price = price.loc[df.index]
        accommodates = accommodates.loc[df.index]

    if budget_min is not None:
        df = df[price >= budget_min]
        base_score = base_score.loc[df.index]
        price = price.loc[df.index]
        accommodates = accommodates.loc[df.index]

    if budget_max is not None:
        df = df[price <= budget_max]
        base_score = base_score.loc[df.index]
        price = price.loc[df.index]
        accommodates = accommodates.loc[df.index]

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # 2) Score de budget (plus le prix est proche du centre, mieux c'est)
    # ------------------------------------------------------------------
    if budget_min is not None and budget_max is not None and budget_max > budget_min:
        target = 0.5 * (budget_min + budget_max)
        # largeur approx : on tolère +- 30% autour du centre
        sigma = 0.3 * (budget_max - budget_min)
        if sigma <= 0:
            sigma = max(target * 0.3, 1.0)

        budget_score = np.exp(-((price - target) ** 2) / (2 * sigma**2))
    else:
        # Pas de budget explicit → on ne pénalise pas
        budget_score = pd.Series(1.0, index=df.index)

    # ------------------------------------------------------------------
    # 3) Score de capacité (>= n_guests est déjà filtré ;
    #    on favorise légèrement quand accommodates est proche de n_guests)
    # ------------------------------------------------------------------
    if n_guests is not None:
        cap_ratio = n_guests / (accommodates + 1e-8)
        cap_score = np.exp(-((cap_ratio - 1.0) ** 2) / 0.5)
    else:
        cap_score = pd.Series(1.0, index=df.index)

    # ------------------------------------------------------------------
    # 4) Normalisation et combinaison
    # ------------------------------------------------------------------
    def _norm(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        min_v, max_v = float(s.min()), float(s.max())
        if max_v - min_v < 1e-8:
            return pd.Series(1.0, index=s.index)
        return (s - min_v) / (max_v - min_v)

    base_norm = _norm(base_score)
    budget_norm = _norm(budget_score)
    cap_norm = _norm(cap_score)

    # Poids : 70% retrieval, 20% budget, 10% capacité
    final_score = 0.7 * base_norm + 0.2 * budget_norm + 0.1 * cap_norm

    df["final_score"] = final_score

    df = df.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Classe ContextualReranker pour offline_eval (compat)
# -------------------------------------------------------------------

@dataclass
class ContextualReranker:
    """
    Version simple de reranker pour l'évaluation offline.
    Elle wrappe rerank_items() pour rester compatible avec le code existant.

    Interface attendue par offline_eval :
        rerank(candidate_ids, base_scores, budget_min, budget_max, n_guests, k_final)
        -> (reordered_ids, reordered_scores)
    """

    def __post_init__(self):
        # On charge les listings pour retrouver les features (price, accommodates, city, etc.)
        self.listings = pd.read_parquet(LISTINGS_V1)

    def rerank(
        self,
        candidate_ids: List[int],
        base_scores: np.ndarray,
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None,
        n_guests: Optional[int] = None,
        k_final: int = 10,
    ) -> Tuple[List[int], np.ndarray]:
        if len(candidate_ids) == 0:
            return [], np.array([])

        cand_df = pd.DataFrame(
            {
                "id": candidate_ids,
                "score": base_scores,
            }
        )

        # On merge avec les features des listings
        cand_df = cand_df.merge(
            self.listings,
            on="id",
            how="left",
            suffixes=("", "_listing"),
        )

        reranked = rerank_items(
            cand_df,
            user_id=None,  # offline_eval ne personnalise pas par user ici
            budget_min=budget_min,
            budget_max=budget_max,
            n_guests=n_guests,
            k=k_final,
        )

        if reranked.empty:
            return [], np.array([])

        new_ids = reranked["id"].astype(int).tolist()
        new_scores = reranked["final_score"].to_numpy(dtype=float)
        return new_ids, new_scores
