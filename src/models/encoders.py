# src/models/encoders.py
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

try:
    import open_clip
except ImportError:  # permet de tourner même sans open_clip
    open_clip = None


class TextEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> torch.Tensor:
        emb = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return emb  # [batch, dim]


class ImageEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        if open_clip is None:
            raise ImportError(
                "open_clip-torch n'est pas installé. Installe-le ou désactive use_image."
            )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: tensor [B, 3, H, W] déjà prétraité
        """
        with torch.no_grad():
            emb = self.model.encode_image(images.to(self.device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb
