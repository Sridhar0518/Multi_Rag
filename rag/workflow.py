"""LangGraph workflow: Query Expansion -> Hybrid Search -> RRF Fusion -> Rerank -> Synthesis.

Any node can be disabled via state['disabled'] (pass-through), which powers the Tab 3 toggles.
"""
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import text_of
from prompts import ANSWER_TMPL, system_prompt
from rag import graph_rag, hyde, standard
from rag.common import RagContext, Timer
from rag.fusion import rerank, rrf

ENGINES = {"Standard RAG": standard, "HyDE RAG": hyde, "Graph RAG": graph_rag}
NODES = ["query_expansion", "hybrid_search", "rrf_fusion", "rerank", "synthesis"]


class RagState(TypedDict, total=False):
    question: str
    practice: str
    practice_filter: str | None
    engine: str
    disabled: list[str]
    disabled_edges: list[str]
    queries: list[str]
    seeds: list[str]
    hyde_passages: list[str]
    rank_lists: list[list[str]]
    graph_edges: list
    fused: list[str]
    final_ids: list[str]
    answer: str
    timings: dict
    llm_tokens: int


def build_graph(ctx: RagContext):
    def off(s, n):
        return n in s.get("disabled", [])

    def n_expand(s):
        t = s.setdefault("timings", {})
        if off(s, "query_expansion"):
            return {"queries": [s["question"]], "seeds": []}
        with Timer(t, "expansion_ms"):
            r = ENGINES[s["engine"]].expand(ctx, s)
        return {**r, "timings": t}

    def cut(s, a, b):
        return f"{a}->{b}" in s.get("disabled_edges", [])

    def n_search(s):
        t = s.setdefault("timings", {})
        if cut(s, "query_expansion", "hybrid_search"):
            s = {**s, "queries": [s["question"]], "seeds": []}
        if off(s, "hybrid_search"):
            return {"rank_lists": [], "graph_edges": []}
        with Timer(t, "retrieval_ms"):
            r = ENGINES[s["engine"]].search(ctx, s)
        return {**r, "timings": t}

    def n_fuse(s):
        lists = [] if cut(s, "hybrid_search", "rrf_fusion") else s.get("rank_lists", [])
        if off(s, "rrf_fusion"):
            fused = lists[0] if lists else []   # no fusion: take first list only
        else:
            fused = [d for d, _ in rrf(lists, ctx.hp.rrf_alpha)]
        return {"fused": fused}

    def n_rerank(s):
        fused = [] if cut(s, "rrf_fusion", "rerank") else s.get("fused", [])
        if off(s, "rerank"):
            return {"final_ids": fused[: ctx.hp.top_k]}
        return {"final_ids": rerank(s["question"], fused, ctx.index.docs, ctx.hp.top_k)}

    def n_synth(s):
        t = s.setdefault("timings", {})
        ids = [] if cut(s, "rerank", "synthesis") else s.get("final_ids", [])
        if off(s, "synthesis"):
            return {"answer": "(synthesis disabled) Retrieved: " + ", ".join(ids), "timings": t}
        srcs = "\n\n".join(f"[S{i+1}] ({ctx.index.docs[c].metadata['source']}) {ctx.index.docs[c].page_content}"
                           for i, c in enumerate(ids))
        if s.get("engine") == "Graph RAG" and s.get("graph_edges"):
            facts = "\n".join(f"- {u} --{r}--> {v}" for u, v, r, _ in s["graph_edges"][:25])
            srcs += "\n\n[Graph relations]\n" + facts
        prompt = ANSWER_TMPL.format(system=system_prompt(s["practice"]),
                                    sources=srcs or "(none retrieved)", question=s["question"])
        with Timer(t, "llm_ms"):
            ans = text_of(ctx.llm.invoke(prompt))
        return {"answer": ans, "timings": t}

    g = StateGraph(RagState)
    for name, fn in zip(NODES, [n_expand, n_search, n_fuse, n_rerank, n_synth]):
        g.add_node(name, fn)
    g.add_edge(START, NODES[0])
    for a, b in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(NODES[-1], END)
    return g.compile()


def run(ctx: RagContext, question: str, engine: str, practice: str,
        disabled: list[str] | None = None, practice_filter: str | None = None,
        disabled_edges: list[str] | None = None) -> dict[str, Any]:
    import time
    app = build_graph(ctx)
    t0 = time.perf_counter()
    out = app.invoke({"question": question, "engine": engine, "practice": practice,
                      "disabled": disabled or [], "disabled_edges": disabled_edges or [], "practice_filter": practice_filter, "timings": {}})
    out["timings"]["total_ms"] = (time.perf_counter() - t0) * 1000
    final = [] if "rerank->synthesis" in (disabled_edges or []) else out.get("final_ids", [])
    out["chunks"] = [ctx.index.docs[c] for c in final]
    return out
