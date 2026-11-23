"""Dense (Chroma, HNSW cosine) + sparse (BM25) indexes over chunks."""
import re
import uuid

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tok(t: str):
    return re.findall(r"\w+", t.lower())


class VectorIndex:
    def __init__(self, embedder, persist_dir: str | None = None):
        self.embedder = embedder
        client = chromadb.PersistentClient(persist_dir) if persist_dir else chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(f"chunks_{uuid.uuid4().hex[:8]}",
                                                   metadata={"hnsw:space": "cosine"})
        self.docs: dict[str, Document] = {}
        self._bm25 = None
        self._order: list[str] = []

    def add(self, chunks: list[Document]):
        new = [c for c in chunks if c.metadata["chunk_id"] not in self.docs]
        if not new:
            return
        ids = [c.metadata["chunk_id"] for c in new]
        self.col.add(ids=ids, documents=[c.page_content for c in new],
                     embeddings=self.embedder.embed_documents([c.page_content for c in new]),
                     metadatas=[{k: str(v) for k, v in c.metadata.items()} for c in new])
        for c in new:
            self.docs[c.metadata["chunk_id"]] = c
        self._order = list(self.docs)
        self._bm25 = BM25Okapi([_tok(self.docs[i].page_content) for i in self._order])

    def dense(self, query: str | None, k: int, practice: str | None = None, vec=None) -> list[str]:
        if not self.docs:
            return []
        vec = vec if vec is not None else self.embedder.embed_query(query)
        where = {"practice": practice} if practice else None
        r = self.col.query(query_embeddings=[vec], n_results=min(k, len(self.docs)), where=where)
        return r["ids"][0]

    def sparse(self, query: str, k: int, practice: str | None = None) -> list[str]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(zip(self._order, scores), key=lambda x: -x[1])
        out = [i for i, s in ranked if s > 0 and (not practice or self.docs[i].metadata.get("practice") == practice)]
        return out[:k]
