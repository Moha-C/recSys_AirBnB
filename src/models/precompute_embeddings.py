# recsys_startup/src/models/precompute_embeddings.py

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import requests
from tqdm import tqdm
from PIL import Image
import open_clip

from src.config import LISTINGS_V1, DATA_PROCESSED_DIR



IMG_CACHE_DIR = DATA_PROCESSED_DIR / "img_cache"
VERSION_FILE = DATA_PROCESSED_DIR / "embeddings_version.txt"
EMBEDDINGS_VERSION = "multimodal_v1"  # permet de forcer un recalcul si on change la logique


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha_text",
        type=float,
        default=0.7,
        help="Poids du texte dans la fusion texte/image (0–1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size pour l'encodage CLIP.",
    )
    parser.add_argument(
        "--max_listings",
        type=int,
        default=0,
        help="Nombre max global de listings à encoder (0 = pas de limite globale).",
    )
    parser.add_argument(
        "--max_per_city",
        type=int,
        default=0,
        help="Nombre max de listings à encoder PAR VILLE (0 = pas de limite par ville).",
    )
    parser.add_argument(
        "--max_download",
        type=int,
        default=100,
        help="Nombre max de nouvelles images à télécharger dans img_cache (par ville).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer le recalcul complet des embeddings (ignore le cache).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Cache / existence
# ----------------------------------------------------------------------


def embeddings_exist() -> bool:
    required = [
        DATA_PROCESSED_DIR / "item_ids.npy",
        DATA_PROCESSED_DIR / "emb_text_clip.npy",
        DATA_PROCESSED_DIR / "emb_image_clip.npy",
        DATA_PROCESSED_DIR / "emb_item_fused.npy",
        DATA_PROCESSED_DIR / "item_context_features.parquet",
        VERSION_FILE,
    ]
    if not all(p.exists() for p in required):
        return False

    try:
        v = VERSION_FILE.read_text(encoding="utf-8").strip()
        return v == EMBEDDINGS_VERSION
    except Exception:
        return False


def write_version_file():
    VERSION_FILE.write_text(EMBEDDINGS_VERSION, encoding="utf-8")


# ----------------------------------------------------------------------
# CLIP loader & encoding
# ----------------------------------------------------------------------


def load_clip():
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, preprocess, tokenizer, device


def encode_texts(
    model, tokenizer, device, texts: List[str], batch_size: int
) -> torch.Tensor:
    all_emb = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Encoding TEXT embeddings", unit="batch"
    ):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ----------------------------------------------------------------------
# IMAGE download + encoding
# ----------------------------------------------------------------------


def url_to_filename(url: str, city: Optional[str] = None) -> Optional[str]:
    """
    Génère un nom de fichier unique par ville pour garantir max_download PAR VILLE.
    Exemple : 'Paris_12345.jpg'
    """
    if not isinstance(url, str) or not url.strip():
        return None

    base = url.split("?", 1)[0]
    fname = os.path.basename(base)
    if not fname:
        return None

    if city:
        return f"{city}_{fname}"
    return fname





def download_image_to_cache(url: str, dest: Path, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return False
        img = Image.open(r.raw).convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True
    except Exception:
        return False


def prepare_image_paths(
    picture_urls: List[str],
    cities: Optional[List[str]],
    max_download: int,
) -> List[Optional[Path]]:
    """
    Retourne une liste de paths locaux (ou None) pour chaque URL.

    - Si fichier déjà en cache → on le réutilise.
    - Si pas en cache et max_download par ville non atteint → on télécharge.
    - Sinon → None.

    max_download est interprété comme: NOMBRE MAX DE NOUVELLES IMAGES PAR VILLE.
    """
    IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cities is None or len(cities) != len(picture_urls):
        # fallback: on considère une seule "ville" globale
        cities = ["__unknown__"] * len(picture_urls)

    local_paths: List[Optional[Path]] = []
    downloads_per_city: dict[str, int] = {}

    print(
        f"Resolving local images in {IMG_CACHE_DIR} "
        f"(max new downloads per city: {max_download})..."
    )

    for url, city in tqdm(
            list(zip(picture_urls, cities)),
            desc="Checking img_cache",
            unit="img",
        ):
            # nom de ville normalisé (clé pour le quota + nom de fichier)
            c = str(city) if city is not None else "__unknown__"

            # nom de fichier unique par ville
            fname = url_to_filename(url, c)
            if fname is None:
                local_paths.append(None)
                continue

            path = IMG_CACHE_DIR / fname
            if path.exists():
                # déjà en cache → on le réutilise sans compter dans le quota de "nouvelles" images
                local_paths.append(path)
                continue

            current_count = downloads_per_city.get(c, 0)

            # quota atteint pour cette ville ?
            if current_count >= max_download:
                local_paths.append(None)
                continue

            ok = download_image_to_cache(url, path)
            if ok:
                downloads_per_city[c] = current_count + 1
                local_paths.append(path)
            else:
                local_paths.append(None)


    print("New images downloaded per city:")
    for city, n in downloads_per_city.items():
        print(f"  {city}: {n}")

    return local_paths



def encode_images(
    model, preprocess, device, local_paths: List[Optional[Path]], batch_size: int
) -> torch.Tensor:
    """
    Encode les images locales avec CLIP.
    Retourne un tensor [N, d]; les entrées sans image gardent un embedding 0.
    """
    # dimension d'embedding image
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        d = model.encode_image(dummy).shape[-1]

    N = len(local_paths)
    img_emb = torch.zeros((N, d), dtype=torch.float32)

    # indices avec image
    indices = [i for i, p in enumerate(local_paths) if p is not None]
    if not indices:
        print("No local images available – image embeddings will be all zeros.")
        return img_emb

    for start in tqdm(
        range(0, len(indices), batch_size),
        desc="Encoding IMAGE embeddings",
        unit="batch",
    ):
        batch_idx = indices[start : start + batch_size]
        imgs = []
        for i in batch_idx:
            path = local_paths[i]
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                # si problème de lecture fichier, on laisse le vecteur à zéro
                imgs.append(None)

        # filtrer None
        keep = [x for x in imgs if x is not None]
        if not keep:
            continue

        batch_tensor = torch.stack([x for x in imgs if x is not None]).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu()

        # assigner aux indices correspondants (en ignorant les None)
        j = 0
        for idx, x in zip(batch_idx, imgs):
            if x is None:
                continue
            img_emb[idx] = emb[j]
            j += 1

    return img_emb


# ----------------------------------------------------------------------
# Context features
# ----------------------------------------------------------------------


def add_context_features():
    print("Computing context features…")
    df = pd.read_parquet(LISTINGS_V1)

    # prix normalisé
    if "price" in df.columns:
        p = pd.to_numeric(df["price"], errors="coerce")
        if p.notna().sum() > 0:
            p = p.fillna(p.median())
            mean = p.mean()
            std = p.std() if p.std() > 0 else 1.0
            df["price_z"] = (p - mean) / std
        else:
            df["price_z"] = 0.0
    else:
        df["price_z"] = 0.0

    # capacité
    if "accommodates" in df.columns:
        df["capacity"] = df["accommodates"].fillna(1).astype(int)
    else:
        df["capacity"] = 1

    # dispo 30 jours : on réutilise ce qui a été calculé dans build_v1_dataset
    # (availability_30_ratio par listing / ville)
    if "availability_30_ratio" in df.columns:
        df["availability_30d"] = df["availability_30_ratio"].fillna(0.5)
    else:
        # fallback neutre
        df["availability_30d"] = 0.5

    # sécurité : colonnes texte
    if "neighbourhood" not in df.columns:
        df["neighbourhood"] = ""
    if "city" not in df.columns:
        df["city"] = ""

    out = DATA_PROCESSED_DIR / "item_context_features.parquet"
    df[
        ["id", "price_z", "capacity", "availability_30d", "neighbourhood", "city"]
    ].rename(columns={"id": "listing_id"}).to_parquet(out, index=False)
    print(f"✅ Saved context features to {out}")



# ----------------------------------------------------------------------
# Main compute
# ----------------------------------------------------------------------


def compute_embeddings(
    alpha_text: float,
    max_listings: int,
    max_per_city: int,
    batch_size: int,
    max_download: int,
    force: bool,
):
    if embeddings_exist() and not force:
        print("✅ Embeddings & context already exist – skipping recomputation.")
        return

    print("Loading listings from:", LISTINGS_V1)
    df = pd.read_parquet(LISTINGS_V1)

    if len(df) == 0:
        raise RuntimeError("❌ listings_v1.parquet is empty.")

    # 1) Limite PAR VILLE si demandé
    if max_per_city > 0 and "city" in df.columns:
        print(f"⚠️ Limiting to at most {max_per_city} listings per city.")
        df = (
            df.sort_values("id")  # ou autre critère si tu veux
            .groupby("city", group_keys=False)
            .head(max_per_city)
        )

    # 2) Limite globale en backup
    if max_listings > 0 and len(df) > max_listings:
        print(
            f"⚠️ Limiting listings to first {max_listings} "
            f"for CLIP encoding (RAM-safety)."
        )
        df = df.head(max_listings)

    print(f"Using {len(df)} listings for CLIP encoding.")


    # Texte
    text_cols = [c for c in ["name", "description", "amenities"] if c in df.columns]

    def fuse_text(row: pd.Series) -> str:
        parts = [str(row[c]) for c in text_cols if pd.notna(row[c])]
        return " ".join(parts)

    print(f"Building text corpus for {len(df)} listings…")
    texts = df.apply(fuse_text, axis=1).tolist()

    # CLIP
    model, preprocess, tokenizer, device = load_clip()

    print("Encoding TEXT embeddings with CLIP…")
    text_emb = encode_texts(model, tokenizer, device, texts, batch_size)

    # Images
    # Images
    if "picture_url" in df.columns:
        picture_urls = df["picture_url"].fillna("").astype(str).tolist()
    else:
        picture_urls = ["" for _ in range(len(df))]

    # On passe aussi la ville pour avoir un quota par ville
    if "city" in df.columns:
        cities = df["city"].fillna("__unknown__").astype(str).tolist()
    else:
        cities = ["__unknown__"] * len(df)

    local_paths = prepare_image_paths(
        picture_urls=picture_urls,
        cities=cities,
        max_download=max_download,
    )
    img_emb = encode_images(model, preprocess, device, local_paths, batch_size)


    # Fusion texte + image
    print("Fusing TEXT + IMAGE embeddings…")
    alpha = float(alpha_text)
    fused = alpha * text_emb + (1.0 - alpha) * img_emb
    fused = fused / fused.norm(dim=-1, keepdim=True)

    item_ids = df["id"].astype(int).values

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED_DIR / "item_ids.npy", item_ids)
    np.save(DATA_PROCESSED_DIR / "emb_text_clip.npy", text_emb.numpy())
    np.save(DATA_PROCESSED_DIR / "emb_image_clip.npy", img_emb.numpy())
    np.save(DATA_PROCESSED_DIR / "emb_item_fused.npy", fused.numpy())
    write_version_file()

    print("✅ Saved embeddings to", DATA_PROCESSED_DIR)


def main():
    args = parse_args()
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    compute_embeddings(
        alpha_text=args.alpha_text,
        max_listings=args.max_listings,
        max_per_city=args.max_per_city,   # 🔹 on ajoute cet argument
        batch_size=args.batch_size,
        max_download=args.max_download,
        force=args.force,
    )

    # features de contexte
    if args.force or not (DATA_PROCESSED_DIR / "item_context_features.parquet").exists():
        add_context_features()


if __name__ == "__main__":
    main()
