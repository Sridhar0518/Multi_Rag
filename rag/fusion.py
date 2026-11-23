"""Reciprocal Rank Fusion and a simple lexical-overlap reranker."""
import re


def rrf(rank_lists: list[list[str]], alpha: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for r, doc in enumerate(lst, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (alpha + r)
    return sorted(scores.items(), key=lambda x: -x[1])


_CE = {}


def _cross_encoder():
    if "m" not in _CE:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _CE["m"] = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _CE["m"] = None
    return _CE["m"]


def rerank(query: str, ids: list[str], docs: dict, top_k: int, use_cross_encoder: bool = True) -> list[str]:
    """Cross-encoder rerank (ms-marco MiniLM via fastembed); falls back to lexical overlap."""
    ce = _cross_encoder() if use_cross_encoder and ids else None
    if ce is not None:
        try:
            scores = list(ce.rerank(query, [docs[i].page_content for i in ids]))
            return [i for _, i in sorted(zip(scores, ids), key=lambda x: -x[0])][:top_k]
        except Exception:
            pass
    return lexical_rerank(query, ids, docs, top_k)


def lexical_rerank(query: str, ids: list[str], docs: dict, top_k: int) -> list[str]:
    """Stable rerank: blend fused order with query-term overlap."""
    qt = set(re.findall(r"\w+", query.lower()))
    def score(pair):
        pos, i = pair
        dt = set(re.findall(r"\w+", docs[i].page_content.lower()))
        return -(len(qt & dt) / (len(qt) or 1)) + pos * 0.02
    return [i for _, i in sorted(enumerate(ids), key=score)][:top_k]
