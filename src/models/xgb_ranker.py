# src/models/xgb_ranker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

from src.config import DATA_PROCESSED_DIR, LISTINGS_V1

# Fichier où sera sauvegardé le modèle
XGB_MODEL_PATH = DATA_PROCESSED_DIR / "xgb_ranker.json"

# Colonnes candidates pour faire des features sur les listings
FEATURE_COLUMNS_CANDIDATES = [
    "price",
    "review_scores_rating",
    "number_of_reviews",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_90",
    "availability_365",
    "cluster_id",
]


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les colonnes utiles (si présentes) + nettoyage.
    """
    cols = [c for c in FEATURE_COLUMNS_CANDIDATES if c in df.columns]
    if not cols:
        # fallback : aucune feature dispo → feature "bias" constante
        return pd.DataFrame({"bias": np.ones(len(df), dtype="float32")}, index=df.index)

    out = df[cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.astype("float32")


@dataclass
class XGBRankerWrapper:
    model: "xgb.XGBClassifier"
    feature_columns: Sequence[str]

    @classmethod
    def load_default(cls) -> Optional["XGBRankerWrapper"]:
        """
        Charge le modèle depuis XGB_MODEL_PATH si possible.
        Retourne None si xgboost pas installé ou modèle absent.
        """
        if xgb is None:
            print("[XGBRanker] xgboost is not installed – skipping.")
            return None
        if not XGB_MODEL_PATH.exists():
            print(f"[XGBRanker] No model found at {XGB_MODEL_PATH}, skipping.")
            return None

        booster = xgb.XGBClassifier()
        booster.load_model(str(XGB_MODEL_PATH))
        feature_columns = FEATURE_COLUMNS_CANDIDATES
        return cls(model=booster, feature_columns=feature_columns)

    def predict_scores(self, cand_df: pd.DataFrame) -> np.ndarray:
        """
        Prend un DataFrame de candidats (listings + colonnes prix, rating, etc.)
        et renvoie un score de "relevance" pour chaque ligne.
        """
        feats = _select_feature_columns(cand_df)

        # Pour XGBClassifier, on utilise predict_proba si dispo
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feats.values)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        else:
            pred = self.model.predict(feats.values)
            return pred.ravel().astype("float32")
