# src/training/aggregate_logs.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.config import DATA_PROCESSED_DIR, AGG_LOGS_PATH

# Les logs UI sont écrits par app.py dans data/processed/logs
LOGS_DIR = DATA_PROCESSED_DIR / "logs"


def aggregate_logs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_files: List[Path] = sorted(LOGS_DIR.glob("interactions_*.jsonl"))
    if not log_files:
        print(f"No log files found in {LOGS_DIR}")
        return

    dfs = []
    for path in log_files:
        print(f"Loading {path}")
        df = pd.read_json(path, lines=True)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # On ne garde que les interactions POSITIVES
    positive_actions = {"click", "like", "thumb_up"}
    df_pos = df_all[df_all["action_type"].isin(positive_actions)].copy()

    if df_pos.empty:
        print("No positive interactions (click/like/thumb_up) found in logs.")
        return

    # Colonne timestamp : on privilégie 'ts' si présente
    if "ts" in df_pos.columns:
        df_pos["timestamp"] = pd.to_datetime(df_pos["ts"], errors="coerce")
    else:
        # fallback : timestamps artificiels
        df_pos["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            df_pos.index, unit="D"
        )

    # City peut ne pas être présente dans tous les logs
    if "city" not in df_pos.columns:
        df_pos["city"] = None

    df_out = df_pos[["user_id", "item_id", "timestamp", "city"]].copy()

    AGG_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(AGG_LOGS_PATH, index=False)
    print(f"✅ Aggregated logs saved to {AGG_LOGS_PATH} ({len(df_out)} interactions)")


if __name__ == "__main__":
    aggregate_logs()
