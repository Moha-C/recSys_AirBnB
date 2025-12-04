# src/training/dataset.py
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairwiseInteractionDataset(Dataset):
    """
    Dataset pour BPR :
    - pour chaque user, on tire (pos_item, neg_item) sous forme d'indices
      dans la matrice d'items (0..n_items-1).
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        item_id_to_idx: Dict[int, int],
        user_min_interactions: int = 2,
    ):
        """
        interactions: DataFrame avec au moins les colonnes ['user_id', 'item_id'].
        item_id_to_idx: mapping {item_id Airbnb -> index 0..n_items-1}
        user_min_interactions: nb min d'items par user pour être gardé.
        """
        self.item_id_to_idx = item_id_to_idx

        # 1) On ne garde que les interactions dont l'item a un embedding
        valid_item_ids = set(item_id_to_idx.keys())
        interactions = interactions[interactions["item_id"].isin(valid_item_ids)].copy()

        # 2) Regroupe par user et remappe chaque item_id -> idx
        user_items: Dict[int, np.ndarray] = {}
        for user_id, group in interactions.groupby("user_id"):
            item_ids = group["item_id"].values
            # remap vers indices
            item_indices = [
                item_id_to_idx[i] for i in item_ids if i in item_id_to_idx
            ]
            if len(item_indices) >= user_min_interactions:
                user_items[user_id] = np.array(item_indices, dtype=np.int64)

        self.user_items = user_items
        self.users: List[int] = list(self.user_items.keys())

        # 3) Liste globale des indices d'items possibles pour le tirage négatif
        self.all_item_indices = np.array(
            list(item_id_to_idx.values()), dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        """
        Retourne:
          - user_id (pour personnalisation future)
          - pos_idx: index d'un item positif (0..n_items-1)
          - neg_idx: index d'un item négatif (0..n_items-1)
        """
        user_id = self.users[idx]
        items = self.user_items[user_id]  # np.array d'indices

        # tirage positif
        pos_idx = np.random.choice(items)

        # tirage négatif (un item que ce user n'a jamais vu)
        neg_idx = np.random.choice(self.all_item_indices)
        # on évite de tirer un item déjà dans les positifs du user
        # (set pour accélérer le "in")
        items_set = set(items.tolist())
        while neg_idx in items_set:
            neg_idx = np.random.choice(self.all_item_indices)

        return user_id, int(pos_idx), int(neg_idx)
