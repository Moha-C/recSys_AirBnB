# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "week3"

LISTINGS_V1 = DATA_PROCESSED_DIR / "listings_v1.parquet"
INTERACTIONS_V1 = DATA_PROCESSED_DIR / "interactions_v1.parquet"



# Embeddings + FAISS (adapte si tes noms sont différents)
EMB_ITEM_FUSED_PATH = DATA_PROCESSED_DIR / "emb_item_fused.npy"
FUSED_EMB_NPY = EMB_ITEM_FUSED_PATH
EMB_ITEM_IDS_PATH = DATA_PROCESSED_DIR / "emb_item_ids.npy"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "faiss_index_fused.bin"

# Logs & utilisateurs
LOGS_DIR = DATA_DIR / "logs"
INTERACTIONS_LOG_PATH = LOGS_DIR / "interactions.jsonl"
AGG_LOGS_PATH = DATA_PROCESSED_DIR / "interactions_agg.parquet"

USERS_PATH = DATA_PROCESSED_DIR / "users.json"

# Training
MODEL_WEIGHTS = DATA_PROCESSED_DIR / "multimodal_model.pt"
LOSS_CURVES_PNG = EXPERIMENTS_DIR / "loss_curves.png"
METRICS_LAST = EXPERIMENTS_DIR / "train_last_metrics.json"
