from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str | Path, batch_size: int = 32, show_progress: bool = False) -> None:
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = SentenceTransformer(self.model_path)

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        ).astype(np.float32)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
