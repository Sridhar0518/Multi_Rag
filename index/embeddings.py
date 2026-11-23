"""bge-large-en-v1.5 (1024-dim) via fastembed/ONNX; deterministic hash fallback for tests."""
import hashlib
import re

import numpy as np

from config import EMBED_MODEL

DIM = 1024


class BgeEmbedder:
    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(EMBED_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge retrieval instruction for queries
        q = "Represent this sentence for searching relevant passages: " + text
        return next(iter(self.model.embed([q]))).tolist()


class HashEmbedder:
    """Offline bag-of-words hashing embedder (tests / no-network fallback)."""
    def _v(self, text):
        v = np.zeros(DIM)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)
