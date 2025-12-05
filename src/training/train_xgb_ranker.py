# src/training/train_xgb_ranker.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.config import LISTINGS_V1, AGG_LOGS_PATH, DATA_PROCESSED_DIR
from src.models.xgb_ranker import (
    _select_feature_columns,
    XGB_MODEL_PATH,
)







def build_training_dataframe(
    logs: pd.DataFrame,
    listings: pd.DataFrame,
    n_negative: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    logs = logs.copy()
    logs = logs.dropna(subset=["user_id", "item_id"])
    logs["user_id"] = logs["user_id"].astype(int)
    logs["item_id"] = logs["item_id"].astype(int)

    # On joint les logs avec listings_v1 pour récupérer les features items
    interactions = logs.merge(
        listings,
        left_on="item_id",
        right_on="id",
        how="inner",
        suffixes=("", "_item"),
    )

    if interactions.empty:
        raise RuntimeError("No interactions left after join with listings_v1.")

    rows: List[dict] = []

    all_item_ids = listings["id"].astype(int).to_numpy()
    by_user = interactions.groupby("user_id")

    for user_id, group in by_user:
        pos_item_ids = group["item_id"].astype(int).to_numpy()
        if len(pos_item_ids) == 0:
            continue

        for _, row in group.iterrows():
            # Échantillon positif
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(row["item_id"]),
                    "label": 1,
                }
            )

            # Échantillons négatifs : items jamais vus par ce user
            possible_neg = np.setdiff1d(all_item_ids, pos_item_ids)
            if possible_neg.size == 0:
                continue

            n_to_sample = min(n_negative, len(possible_neg))
            neg_sample = rng.choice(
                possible_neg,
                size=n_to_sample,
                replace=False,
            ).astype(int)

            for neg_item in neg_sample:
                rows.append(
                    {
                        "user_id": int(user_id),
                        "item_id": int(neg_item),
                        "label": 0,
                    }
                )

    df_train = pd.DataFrame(rows)
    if df_train.empty:
        raise RuntimeError("Training dataframe is empty (no positive/negative pairs built).")

    # On rejoint encore une fois pour avoir toutes les features des items
    df_train = df_train.merge(
        listings,
        left_on="item_id",
        right_on="id",
        how="left",
    )

    return df_train


def main():
    if xgb is None:
        raise RuntimeError(
            "xgboost is not installed. Install it with `pip install xgboost` in your venv."
        )

    if not AGG_LOGS_PATH.exists():
        raise RuntimeError(
            f"No aggregated logs found at {AGG_LOGS_PATH}. "
            "Run `python -m src.training.aggregate_logs` first."
        )

    logs = pd.read_parquet(AGG_LOGS_PATH)
    listings = pd.read_parquet(LISTINGS_V1)

    df_train = build_training_dataframe(logs, listings, n_negative=5)

    X = _select_feature_columns(df_train)
    y = df_train["label"].astype(int).to_numpy()

    # Petit train/val split simple
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(XGB_MODEL_PATH))
    print(f"Saved XGBoost ranker model to {XGB_MODEL_PATH}")


if __name__ == "__main__":
    main()
