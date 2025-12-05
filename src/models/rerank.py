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
from src.models.xgb_ranker import XGBRankerWrapper


@dataclass
class ContextualReranker:
    """
    Stage 2 reranker.

    - Prend une liste de candidats (ids + scores CLIP de base)
    - Applique des filtres (budget, guests)
    - Calcule un score manuel simple
    - Optionnellement : ajuste le score avec un XGBoost entraîné sur les logs
    """

    listings_path: Path = LISTINGS_V1

    def __post_init__(self) -> None:
        # Chargement unique des listings
        self.listings = pd.read_parquet(self.listings_path)
        # XGBoost ranker optionnel (None si pas de modèle ou pas de xgboost)
        self.xgb_ranker = XGBRankerWrapper.load_default()

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _build_candidate_frame(self, candidate_ids: np.ndarray, base_scores: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "id": candidate_ids.astype(int),
                "base_score": base_scores.astype("float32"),
            }
        )
        df = df.merge(self.listings, on="id", how="left")
        return df

    def _apply_budget_filter(
        self,
        df: pd.DataFrame,
        budget_min: Optional[float],
        budget_max: Optional[float],
    ) -> pd.DataFrame:
        if "price" not in df.columns:
            return df

        out = df.copy()
        if budget_min is not None:
            out = out[out["price"] >= float(budget_min)]
        if budget_max is not None and budget_max > 0:
            out = out[out["price"] <= float(budget_max)]
        return out

    def _apply_guests_filter(
        self,
        df: pd.DataFrame,
        n_guests: Optional[int],
    ) -> pd.DataFrame:
        if n_guests is None:
            return df
        if "accommodates" not in df.columns:
            return df
        out = df.copy()
        out = out[out["accommodates"] >= int(n_guests)]
        return out

    def _compute_final_score_without_xgb(self, df: pd.DataFrame) -> pd.Series:
        """
        Score "manuel" : combinaison de base_score + signaux simples.
        """
        score = df["base_score"].astype("float32").copy()

        # Bonus léger pour les biens mieux notés
        if "review_scores_rating" in df.columns:
            rating = df["review_scores_rating"].fillna(0) / 100.0
            score = score + 0.05 * rating.astype("float32")

        # Pénalisation légère pour les prix très anormaux
        if "price" in df.columns:
            price = df["price"].fillna(df["price"].median())
            price_norm = (price - price.mean()) / (price.std() + 1e-6)
            score = score - 0.01 * price_norm.astype("float32")

        return score

    # ------------------------------------------------------------------
    # API principale (utilisée par offline_eval + potentiellement l'API)
    # ------------------------------------------------------------------
    def rerank(
        self,
        candidate_ids: np.ndarray,
        base_scores: np.ndarray,
        budget_min: Optional[float],
        budget_max: Optional[float],
        n_guests: Optional[int],
        k: int,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Retourne (new_ids, new_scores) après reranking.
        """
        if candidate_ids.size == 0:
            return [], np.array([], dtype="float32")

        cand_df = self._build_candidate_frame(candidate_ids, base_scores)

        # Filtres budget / guests
        cand_df = self._apply_budget_filter(cand_df, budget_min, budget_max)
        cand_df = self._apply_guests_filter(cand_df, n_guests)

        if cand_df.empty:
            return [], np.array([], dtype="float32")

        # Score manuel de base
        cand_df["final_score"] = self._compute_final_score_without_xgb(cand_df)

        # Ajout éventuel du XGBoost ranker
        if self.xgb_ranker is not None:
            try:
                xgb_scores = self.xgb_ranker.predict_scores(cand_df)
                # Normalisation pour éviter une échelle bizarre
                xgb_scores = (xgb_scores - xgb_scores.mean()) / (xgb_scores.std() + 1e-6)
                base = cand_df["final_score"].to_numpy(dtype="float32")
                # Combinaison 50/50 entre manuel et XGB
                final = 0.5 * base + 0.5 * xgb_scores.astype("float32")
                cand_df["final_score"] = final
            except Exception as e:  # pragma: no cover
                print(f"[XGBRanker] Error while applying model: {e}. Falling back to manual scores.")

        # Tri décroissant + top-k
        cand_df = cand_df.sort_values("final_score", ascending=False).head(k)

        new_ids = cand_df["id"].astype(int).tolist()
        new_scores = cand_df["final_score"].to_numpy(dtype="float32")
        return new_ids, new_scores
