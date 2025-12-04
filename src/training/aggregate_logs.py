# src/data/aggregate_logs.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.config import DATA_PROCESSED_DIR

LOGS_DIR = DATA_PROCESSED_DIR / "logs"
OUT_PATH = DATA_PROCESSED_DIR / "interactions_from_logs.parquet"


def aggregate_logs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    files: List[Path] = sorted(LOGS_DIR.glob("interactions_*.jsonl"))
    if not files:
        print("No log files found in", LOGS_DIR)
        return

    parts = []
    for path in files:
        print("Loading", path)
        df = pd.read_json(path, lines=True)
        parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)

    # On garde seulement les événements positifs pour l'entraînement
    df_pos = df_all[df_all["action_type"].isin(["click", "like"])].copy()

    if df_pos.empty:
        print("No positive interactions (click/like) found in logs.")
        return

    # Normalisation des colonnes pour coller à INTERACTIONS_V1
    df_pos["user_id"] = df_pos["user_id"].astype(int)
    df_pos["item_id"] = df_pos["item_id"].astype(int)
    if "city" not in df_pos.columns:
        df_pos["city"] = None

    if "ts" in df_pos.columns:
        df_pos["timestamp"] = pd.to_datetime(df_pos["ts"], errors="coerce")
    else:
        df_pos["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            df_pos.index, unit="D"
        )

    df_out = df_pos[["user_id", "item_id", "timestamp", "city"]].copy()
    df_out.to_parquet(OUT_PATH, index=False)
    print(f"✅ Aggregated logs saved to {OUT_PATH} ({len(df_out)} interactions)")


if __name__ == "__main__":
    aggregate_logs()
