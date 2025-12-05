# src/data/build_v1_dataset.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, LISTINGS_V1, INTERACTIONS_V1

# ---------------------------------------------------------------------------
# Imports optionnels pour le clustering géographique
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Colonnes minimales utilisées par le reste du pipeline
BASE_LISTING_COLS = [
    "id",
    "name",
    "description",
    "amenities",
    "neighbourhood",
    "latitude",
    "longitude",
    "price",
    "picture_url",
    "accommodates",
    "city",
]

# Colonnes enrichies (calendar + géo)
EXTRA_LISTING_COLS = [
    "calendar_price_mean",
    "calendar_price_median",
    "calendar_price_min",
    "calendar_price_max",
    "occupancy_rate",
    "availability_30_ratio",
    "neighbourhood_lat",
    "neighbourhood_lon",
    "geo_cluster",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cities",
        type=str,
        default="Paris,Lyon,Bordeaux",
        help="Liste de villes à charger depuis data/raw/<city>/...",
    )
    parser.add_argument(
        "--min_reviews",
        type=int,
        default=3,
        help="Nombre minimum de reviews pour garder un listing (si la colonne existe).",
    )
    return parser.parse_args()


# ===========================================================================
# Utils génériques
# ===========================================================================

def _city_dir(city: str) -> Path:
    return DATA_RAW_DIR / city


def _find_existing(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


# ===========================================================================
# Chargement listings
# ===========================================================================

def load_city_listings(city: str) -> pd.DataFrame:
    """
    Charge les listings d'une ville :
    - data/raw/<city>/listings.csv.gz
    - data/raw/<city>/listings.csv
    Sinon, fallback data/raw/listings.csv.gz (legacy).
    """
    city_dir = _city_dir(city)
    candidates = [
        city_dir / "listings.csv.gz",
        city_dir / "listings.csv",
    ]
    path = _find_existing(candidates)

    if path is None:
        legacy = DATA_RAW_DIR / "listings.csv.gz"
        if not legacy.exists():
            raise FileNotFoundError(
                f"Aucun listings.csv trouvé pour {city} (ni {city_dir}/listings.* ni {legacy})."
            )
        print(f"[{city}] WARNING: using legacy {legacy}, city='{city}' for all rows.")
        df = pd.read_csv(legacy, low_memory=False)
        df["city"] = city
        return df

    print(f"[{city}] Loading listings from {path}")
    df = pd.read_csv(path, low_memory=False)
    df["city"] = city
    return df


# ===========================================================================
# Chargement reviews
# ===========================================================================

def load_city_reviews(city: str) -> pd.DataFrame:
    """
    Reviews pour interactions :
    - data/raw/<city>/reviews.csv(.gz)
    - sinon data/raw/reviews.csv(.gz) (global)
    - sinon DataFrame vide
    """
    city_dir = _city_dir(city)
    candidates = [
        city_dir / "reviews.csv.gz",
        city_dir / "reviews.csv",
    ]
    path = _find_existing(candidates)
    if path is not None:
        print(f"[{city}] Loading reviews from {path}")
        df = pd.read_csv(path, low_memory=False)
        df["city"] = city
        return df

    global_candidates = [
        DATA_RAW_DIR / "reviews.csv.gz",
        DATA_RAW_DIR / "reviews.csv",
    ]
    path = _find_existing(global_candidates)
    if path is not None:
        print(
            f"[{city}] WARNING: using global reviews file {path}, "
            "city will be inferred by join with listings_v1."
        )
        df = pd.read_csv(path, low_memory=False)
        df["city"] = None
        return df

    print(f"[{city}] WARNING: no reviews file found – interactions will be empty for this city.")
    return pd.DataFrame()


# ===========================================================================
# Chargement calendar (prix / dispo)
# ===========================================================================

def load_city_calendar(city: str) -> pd.DataFrame:
    """
    Calendar pour prix & disponibilité :
    - data/raw/<city>/calendar.csv(.gz)
    - sinon DataFrame vide.
    """
    city_dir = _city_dir(city)
    candidates = [
        city_dir / "calendar.csv.gz",
        city_dir / "calendar.csv",
    ]
    path = _find_existing(candidates)
    if path is None:
        print(f"[{city}] WARNING: no calendar.csv found – no calendar features.")
        return pd.DataFrame()

    print(f"[{city}] Loading calendar from {path}")
    cal = pd.read_csv(path, low_memory=False)
    cal["city"] = city
    return cal


def build_calendar_features(city: str) -> pd.DataFrame:
    cal = load_city_calendar(city)
    if cal.empty:
        return pd.DataFrame(
            columns=[
                "listing_id",
                "calendar_price_mean",
                "calendar_price_median",
                "calendar_price_min",
                "calendar_price_max",
                "occupancy_rate",
                "availability_30_ratio",
                "city",
            ]
        )

    # Colonnes attendues typiques InsideAirbnb: listing_id, date, available, price
    if "listing_id" not in cal.columns:
        raise ValueError(f"[{city}] calendar.csv must have a 'listing_id' column.")

    # Nettoyage des prix (string -> float)
    if "price" in cal.columns:
        cal["price"] = (
            cal["price"]
            .astype(str)
            .str.replace(r"[^0-9.,]", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        cal["price"] = pd.to_numeric(cal["price"], errors="coerce")
    else:
        cal["price"] = np.nan

    # Date
    if "date" in cal.columns:
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    else:
        cal["date"] = pd.NaT

    # Available: 't' / 'f'
    if "available" in cal.columns:
        available_flag = cal["available"].astype(str).str.lower().str.startswith("t")
    else:
        available_flag = pd.Series(False, index=cal.index)

    # Agrégations globales
    grp = cal.groupby("listing_id", as_index=False)
    agg = grp.agg(
        calendar_price_mean=("price", "mean"),
        calendar_price_median=("price", "median"),
        calendar_price_min=("price", "min"),
        calendar_price_max=("price", "max"),
        days_total=("date", "count"),
        days_available=("available", lambda s: (s.astype(str).str.lower().str.startswith("t")).sum())
        if "available" in cal.columns
        else ("date", "count"),
    )

    agg["occupancy_rate"] = 1.0 - (agg["days_available"] / agg["days_total"]).replace(
        {0: np.nan}
    )

    # Dispo sur 30 jours : on prend la fenêtre min(date)->min(date)+30
    if cal["date"].notna().any():
        start = cal["date"].min()
        end = start + pd.Timedelta(days=30)
        cal_30 = cal[cal["date"].between(start, end)]
        grp_30 = cal_30.groupby("listing_id")
        avail_30 = grp_30["available"].apply(
            lambda s: (s.astype(str).str.lower().str.startswith("t")).sum()
        )
        days_30 = grp_30["date"].count()
        availability_30_ratio = (avail_30 / days_30).replace({0: np.nan})
        agg = agg.merge(
            availability_30_ratio.rename("availability_30_ratio"),
            left_on="listing_id",
            right_index=True,
            how="left",
        )
    else:
        agg["availability_30_ratio"] = np.nan

    agg["city"] = city
    return agg[
        [
            "listing_id",
            "calendar_price_mean",
            "calendar_price_median",
            "calendar_price_min",
            "calendar_price_max",
            "occupancy_rate",
            "availability_30_ratio",
            "city",
        ]
    ]


# ===========================================================================
# Chargement neighbourhoods.geojson
# ===========================================================================

def build_neighbourhood_features(city: str) -> pd.DataFrame:
    """
    Lit data/raw/<city>/neighbourhoods.geojson (ou neighborhoods.geojson),
    calcule un centroid (lon, lat) par quartier.
    """
    city_dir = _city_dir(city)
    candidates = [
        city_dir / "neighbourhoods.geojson",
        city_dir / "neighborhoods.geojson",
    ]
    path = _find_existing(candidates)
    if path is None:
        print(f"[{city}] WARNING: no neighbourhoods.geojson found – no geo centroids.")
        return pd.DataFrame(
            columns=["neighbourhood", "neighbourhood_lat", "neighbourhood_lon", "city"]
        )

    print(f"[{city}] Loading neighbourhoods from {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    rows: List[Dict[str, Any]] = []

    def collect_coords(coords, out):
        # coords peut être [ [lon,lat], ... ] ou des niveaux imbriqués
        if not coords:
            return
        first = coords[0]
        if isinstance(first, (float, int)):
            # Une coord seule -> on ne traite pas ce cas (peu probable)
            return
        if isinstance(first[0], (float, int)):
            # Liste de [lon,lat] ou anneau
            for c in coords:
                if len(c) >= 2:
                    out.append((c[0], c[1]))
        else:
            # Imbrication (MultiPolygon, etc.)
            for sub in coords:
                collect_coords(sub, out)

    for feat in features:
        props = feat.get("properties", {}) or {}
        name = (
            props.get("neighbourhood")
            or props.get("neighborhood")
            or props.get("name")
        )
        if not name:
            continue

        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [])
        pts: List[tuple] = []
        collect_coords(coords, pts)

        if pts:
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            lon = float(np.mean(lons))
            lat = float(np.mean(lats))
        else:
            lon = np.nan
            lat = np.nan

        rows.append(
            {
                "neighbourhood": name,
                "neighbourhood_lat": lat,
                "neighbourhood_lon": lon,
                "city": city,
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# Construction listings_v1
# ===========================================================================

def build_listings(cities: List[str], min_reviews: int) -> pd.DataFrame:
    all_listings = []
    all_calendar_features = []
    all_neigh_features = []

    for city in cities:
        # listings bruts
        raw = load_city_listings(city)

        # calendar features
        cal_feat = build_calendar_features(city)
        all_calendar_features.append(cal_feat)

        # neighbourhoods geojson -> centroids
        neigh_feat = build_neighbourhood_features(city)
        all_neigh_features.append(neigh_feat)

        # Normalisation colonnes de base
        for col in BASE_LISTING_COLS:
            if col not in raw.columns and col != "city":
                raw[col] = np.nan

        # On garde les colonnes de base + info de reviews si dispo
        base_cols = BASE_LISTING_COLS.copy()
        if "number_of_reviews" in raw.columns:
            base_cols.append("number_of_reviews")

        raw = raw[base_cols].copy()

        # Filtre min_reviews
        if "number_of_reviews" in raw.columns:
            before = len(raw)
            raw = raw[raw["number_of_reviews"].fillna(0) >= min_reviews]
            after = len(raw)
            print(f"[{city}] Filter min_reviews>={min_reviews}: {before} -> {after}")
            raw = raw.drop(columns=["number_of_reviews"])

        all_listings.append(raw)

    # Merge multi-villes
    listings = pd.concat(all_listings, ignore_index=True)

    # Nettoyage prix
    if listings["price"].dtype == object:
        listings["price"] = (
            listings["price"]
            .astype(str)
            .str.replace(r"[^0-9.,]", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        listings["price"] = pd.to_numeric(listings["price"], errors="coerce")

    # Id obligatoire
    listings = listings.dropna(subset=["id"]).copy()
    listings["id"] = listings["id"].astype(int)

    # Calendar features (multi-villes)
    cal_all = pd.concat(all_calendar_features, ignore_index=True) if all_calendar_features else pd.DataFrame()
    if not cal_all.empty:
        cal_all = cal_all.rename(columns={"listing_id": "id"})
        listings = listings.merge(
            cal_all,
            on=["id", "city"],
            how="left",
        )
    else:
        for col in [
            "calendar_price_mean",
            "calendar_price_median",
            "calendar_price_min",
            "calendar_price_max",
            "occupancy_rate",
            "availability_30_ratio",
        ]:
            listings[col] = np.nan

    # Neighbourhood geo centroids
    neigh_all = pd.concat(all_neigh_features, ignore_index=True) if all_neigh_features else pd.DataFrame()
    if not neigh_all.empty:
        listings = listings.merge(
            neigh_all,
            on=["neighbourhood", "city"],
            how="left",
        )
    else:
        listings["neighbourhood_lat"] = np.nan
        listings["neighbourhood_lon"] = np.nan

    # Clustering géographique (par ville)
    listings = add_geo_clusters(listings)

    # Assurer la présence de toutes les colonnes attendues
    for col in BASE_LISTING_COLS + EXTRA_LISTING_COLS:
        if col not in listings.columns:
            listings[col] = np.nan

    # Ordonner les colonnes
    keep_cols = BASE_LISTING_COLS + EXTRA_LISTING_COLS
    other_cols = [c for c in listings.columns if c not in keep_cols]
    listings = listings[keep_cols + other_cols]

    print(f"Listings_v1 final shape: {listings.shape}")
    return listings


def add_geo_clusters(listings: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """
    Cluster géographique simple par ville sur (lat, lon).
    Si scikit-learn n'est pas dispo, on crée geo_cluster=-1.
    """
    if not HAS_SKLEARN:
        print("WARNING: scikit-learn not installed – geo clustering skipped (geo_cluster=-1).")
        listings["geo_cluster"] = -1
        return listings

    listings = listings.copy()
    listings["geo_cluster"] = -1

    for city in listings["city"].dropna().unique():
        mask = listings["city"] == city
        coords = listings.loc[mask, ["latitude", "longitude"]].dropna()
        if len(coords) < n_clusters:
            # Pas assez de points -> on saute
            continue

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(coords.to_numpy())

        # On doit ré-affecter les labels aux lignes correspondantes
        idx_coords = coords.index
        listings.loc[idx_coords, "geo_cluster"] = labels

    return listings


# ===========================================================================
# Construction interactions_v1 à partir des reviews
# ===========================================================================

def build_interactions(listings_v1: pd.DataFrame, cities: List[str]) -> pd.DataFrame:
    """
    Schéma final:
      - user_id
      - item_id
      - timestamp
      - city
    """
    all_reviews = []

    for city in cities:
        df = load_city_reviews(city)
        if df.empty:
            continue
        df["city"] = df["city"].fillna(city)
        all_reviews.append(df)

    if not all_reviews:
        print("No reviews found for any city – interactions_v1 will be empty.")
        return pd.DataFrame(columns=["user_id", "item_id", "timestamp", "city"])

    reviews = pd.concat(all_reviews, ignore_index=True)

    # user_id
# user_id
# user_id
    if "reviewer_id" in reviews.columns:
        reviews["user_id"] = reviews["reviewer_id"]
    elif "reviewer_name" in reviews.columns:
        print("WARNING: using reviewer_name as proxy for user_id.")
        reviews["user_id"] = (
            reviews["reviewer_name"]
            .astype("category")
            .cat.codes.astype(int) + 1
        )
    else:
        print("WARNING: no reviewer_id/reviewer_name – synthetic user_id from index.")
        reviews["user_id"] = reviews.index.astype(int) + 1



    # item_id
    if "listing_id" in reviews.columns:
        reviews["item_id"] = reviews["listing_id"]
    elif "id" in reviews.columns:
        reviews["item_id"] = reviews["id"]
    else:
        raise ValueError("No column 'listing_id' or 'id' in reviews to map to item_id.")

    # timestamp
    if "date" in reviews.columns:
        reviews["timestamp"] = pd.to_datetime(reviews["date"], errors="coerce")
    else:
        reviews["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            reviews.index, unit="D"
        )

    # Join avec listings_v1 pour garder seulement les items existants
    list_ids = listings_v1[["id", "city"]].rename(columns={"id": "item_id"})
    interactions = reviews.merge(list_ids, on=["item_id", "city"], how="inner")

    interactions = interactions[["user_id", "item_id", "timestamp", "city"]].copy()
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["item_id"] = interactions["item_id"].astype(int)

    print(f"Interactions_v1 final shape: {interactions.shape}")
    return interactions


# ===========================================================================
# Main
# ===========================================================================

def main(args: argparse.Namespace):
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    print(f"Building V1 dataset for cities: {cities}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Listings multi-villes enrichis (calendar + géo)
    listings_v1 = build_listings(cities, min_reviews=args.min_reviews)
    listings_v1.to_parquet(LISTINGS_V1, index=False)
    print(f"Saved listings_v1 to {LISTINGS_V1}")

    # 2) Interactions à partir des reviews
    interactions_v1 = build_interactions(listings_v1, cities)
    interactions_v1.to_parquet(INTERACTIONS_V1, index=False)
    print(f"Saved interactions_v1 to {INTERACTIONS_V1}")

    print("✅ build_v1_dataset (Option C: calendar + geojson + clustering) done.")


if __name__ == "__main__":
    main(parse_args())
