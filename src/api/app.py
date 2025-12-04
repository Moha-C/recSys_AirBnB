# src/api/app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal, Any

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import DATA_PROCESSED_DIR, LISTINGS_V1
from src.models.retrieval import search_items
from src.models.rerank import rerank_items

# ---------------------------------------------------------------------
# Chemins pour users & logs
# ---------------------------------------------------------------------
USERS_FILE = DATA_PROCESSED_DIR / "users.json"
LOGS_DIR = DATA_PROCESSED_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Modèles Pydantic (alignés avec le frontend React)
# ---------------------------------------------------------------------
class LoginRequest(BaseModel):
    # le frontend envoie { user_id, password }
    user_id: str
    password: str


class LoginResponse(BaseModel):
    # renvoyé au frontend : authUser = { user_id, display_name, is_new }
    user_id: int            # ID numérique interne
    display_name: str       # label affiché dans le header React
    is_new: bool


class InteractionEvent(BaseModel):
    user_id: Optional[int] = None
    item_id: int
    action_type: Literal["click", "like", "dislike", "hide", "thumb_up", "thumb_down"]
    session_id: Optional[str] = None
    city: Optional[str] = None
    metadata: Optional[dict] = None  # {query, city, n_guests, budget_min, budget_max}
    ts: Optional[datetime] = None


class RecommendItem(BaseModel):
    listing_id: int
    name: str
    description: Optional[str] = None
    neighbourhood: Optional[str] = None
    city: Optional[str] = None
    price: Optional[float] = None
    accommodates: Optional[int] = None
    picture_url: Optional[str] = None
    score: float


class RecommendResponse(BaseModel):
    results: List[RecommendItem]


# ---------------------------------------------------------------------
# Helpers: users (simple fichier JSON)
# ---------------------------------------------------------------------
def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {"next_id": 1, "users": {}}
    with USERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(data: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_or_create_user(login_id: str, password: str) -> LoginResponse:
    """
    login_id = ce que le frontend appelle "user_id" (ex: "alice").
    On le considère comme username, mais on renvoie un user_id numérique.
    """
    login_id = login_id.strip()
    if not login_id:
        raise HTTPException(status_code=400, detail="Empty user_id")

    data = _load_users()
    users = data.get("users", {})
    next_id = int(data.get("next_id", 1))

    if login_id in users:
        u = users[login_id]
        if u["password"] != password:
            raise HTTPException(status_code=401, detail="Wrong password")
        return LoginResponse(
            user_id=u["user_id"],
            display_name=login_id,
            is_new=False,
        )

    # Création d'un nouveau user
    user_id = next_id
    users[login_id] = {
        "user_id": user_id,
        "password": password,  # pour un vrai projet → hacher le mot de passe
    }
    data["users"] = users
    data["next_id"] = user_id + 1
    _save_users(data)

    return LoginResponse(
        user_id=user_id,
        display_name=login_id,
        is_new=True,
    )


# ---------------------------------------------------------------------
# Helpers: logging interactions
# ---------------------------------------------------------------------
def _append_log(event: dict) -> None:
    """
    Écrit l’événement (dict) dans un fichier JSONL du jour :
    data/processed/logs/interactions_YYYYMMDD.jsonl
    """
    # on s’assure d’avoir un timestamp ISO
    if "ts" not in event or event["ts"] is None:
        event["ts"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    fname = LOGS_DIR / f"interactions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    with fname.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Chargement listings (optionnel : pour fallback, pas strictement utilisé ici)
# ---------------------------------------------------------------------
if LISTINGS_V1.exists():
    LISTINGS_DF = pd.read_parquet(LISTINGS_V1)
else:
    LISTINGS_DF = pd.DataFrame()


# ---------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------
app = FastAPI(title="Trip Recommender API", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pour la démo : on ouvre à tout
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# Auth simple (compatible TripRecommender.js)
# ---------------------------------------------------------------------
@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    """
    Auth très simple:
    - si user_id (login_id) existe → on vérifie le password
    - sinon → on crée l'utilisateur
    Le frontend envoie { user_id, password } et récupère { user_id (int), display_name }.
    """
    return get_or_create_user(payload.user_id, payload.password)


# ---------------------------------------------------------------------
# Endpoint de recommandation (compatible TripRecommender.js)
# ---------------------------------------------------------------------
@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    query: str,
    k: int = 10,
    n_guests: int = 2,
    budget_min: float = 0.0,
    budget_max: float = 1e9,
    city: str = "Any",
    user_id: int = 0,
):
    """
    Recommandation principale:
    1. retrieval CLIP (texte+image) via search_items
    2. rerank (budget + calendar + geo + profil user) via rerank_items
    Retourne { results: [...] } pour coller à TripRecommender.js.
    """
    # 1) retrieval
    cand_df = search_items(
        query=query,
        k=max(k * 5, k),
        city=None if city in ("Any", "", "all") else city,
        n_guests=n_guests,
        budget_min=budget_min,
        budget_max=budget_max,
    )

    if cand_df.empty:
        return RecommendResponse(results=[])

    # 2) rerank enrichi
# 2) rerank enrichi
    reranked = rerank_items(
        candidates=cand_df,
        user_id=user_id if user_id != 0 else None,
        budget_min=budget_min,
        budget_max=budget_max,
        n_guests=n_guests,
        k=k,  # <- on passe enfin le k venant du frontend
    )


    #reranked = reranked.head(k)

    results: List[RecommendItem] = []
    for _, row in reranked.iterrows():
        price_val = row.get("price")
        if pd.isna(price_val):
            price = None
        else:
            try:
                price = float(price_val)
            except (TypeError, ValueError):
                price = None

        results.append(
            RecommendItem(
                listing_id=int(row["id"]),
                name=str(row.get("name", "")),
                description=row.get("description"),
                neighbourhood=row.get("neighbourhood"),
                city=row.get("city"),
                price=price,
                accommodates=int(row["accommodates"])
                if not pd.isna(row.get("accommodates"))
                else None,
                picture_url=row.get("picture_url"),
                score=float(row.get("final_score", row.get("score", 0.0))),
            )
        )

    return RecommendResponse(results=results)


# ---------------------------------------------------------------------
# Endpoint logging interactions (compatible TripRecommender.js)
# ---------------------------------------------------------------------
@app.post("/log_interaction")
def log_interaction_endpoint(evt: InteractionEvent):
    """
    Stocke un événement utilisateur dans data/processed/logs/*.jsonl
    - TripRecommender.js envoie :
        { user_id, item_id, action_type, session_id, metadata: {...} }
    """
    try:
        event_dict: dict[str, Any] = evt.model_dump()
        # On peut aplatir quelques infos utiles du metadata si besoin
        meta = event_dict.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("query", "city", "n_guests", "budget_min", "budget_max"):
                if key in meta and key not in event_dict:
                    event_dict[key] = meta[key]

        _append_log(event_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}
