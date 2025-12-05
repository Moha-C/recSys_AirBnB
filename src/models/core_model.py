# src/models/core_model.py
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ModelConfig:
    embedding_dim: int = 384  # dim de all-MiniLM-L6-v2
    alpha_text: float = 0.4   # pondération texte vs image
    use_image: bool = False   # True plus tard


class MultimodalRecModel(nn.Module):
    """
    Two-tower retrieval modèle simple :
    - tower item : fusion texte + image -> proj
    - tower user : moyenne des embeddings des items vus -> proj
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.embedding_dim
        self.item_proj = nn.Linear(d, d)
        self.user_proj = nn.Linear(d, d)

    def encode_item(self, item_emb_text: torch.Tensor) -> torch.Tensor:
        """
        item_emb_text: [N_items, dim]
        (pour Week 3, on ignore l'image ; plus tard on fusionnera)
        """
        x = self.item_proj(item_emb_text)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def encode_user(self, user_item_embs: torch.Tensor) -> torch.Tensor:
        """
        user_item_embs: [N_items_user, dim]
        """
        if user_item_embs.ndim == 2:
            mean_emb = user_item_embs.mean(dim=0, keepdim=True)
        else:
            raise ValueError("user_item_embs doit être [N_items_user, dim]")
        u = self.user_proj(mean_emb)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
        return u  # [1, dim]

    def score(self, user_emb: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        user_emb: [1, dim]
        item_embs: [N_items, dim]
        retourne scores [N_items]
        """
        scores = torch.matmul(item_embs, user_emb.t()).squeeze(-1)
        return scores

    def forward(
        self, user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Pour entraînement BPR / ranking.
        user_emb: [B, dim]
        pos_item_emb: [B, dim]
        neg_item_emb: [B, dim]
        """
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss
