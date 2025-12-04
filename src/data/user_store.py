# src/data/user_store.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import uuid

from src.config import USERS_PATH, DATA_PROCESSED_DIR


def _ensure_dir():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text("{}", encoding="utf-8")


def _load_users() -> Dict[str, Any]:
    _ensure_dir()
    with USERS_PATH.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    return data


def _save_users(data: Dict[str, Any]) -> None:
    _ensure_dir()
    with USERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> Dict[str, Any]:
    """
    Crée un nouvel utilisateur simple (username + mot de passe hashé).
    Retourne {user_id, username}.
    """
    users = _load_users()

    # Vérifier si username déjà pris
    for uid, u in users.items():
        if u.get("username") == username:
            raise ValueError("Username already exists")

    user_id = str(uuid.uuid4())
    users[user_id] = {
        "username": username,
        "password_hash": _hash_password(password),
    }
    _save_users(users)
    return {"user_id": user_id, "username": username}


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    phash = _hash_password(password)

    for uid, u in users.items():
        if u.get("username") == username and u.get("password_hash") == phash:
            return {"user_id": uid, "username": username}
    return None


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    u = users.get(user_id)
    if not u:
        return None
    return {"user_id": user_id, "username": u.get("username")}
