import json

import streamlit as st

from config import CACHE_DIR
from ui.state import get_ctx


def render():
    ctx = get_ctx()
    hp = ctx.hp
    st.subheader("Hyperparameters")
    c = st.columns(4)
    hp.chunk_size = c[0].number_input("Chunk size (tokens)", 100, 1500, hp.chunk_size, 50)
    hp.chunk_overlap = c[0].number_input("Chunk overlap", 0, 300, hp.chunk_overlap, 10)
    hp.hyde_n = c[1].slider("Multi-HyDE passages (N)", 1, 5, hp.hyde_n)
    hp.hop_k = c[2].slider("Graph hop radius (k)", 1, 4, hp.hop_k)
    hp.rrf_alpha = c[3].number_input("RRF constant (alpha)", 1, 200, hp.rrf_alpha)
    hp.top_k = st.slider("Final context size (top-k)", 1, 10, hp.top_k)
    st.caption("Chunk size/overlap apply to documents ingested after the change. Other values apply to the next query.")
    st.subheader("KPIs (last query)")
    last = st.session_state.last
    k = st.columns(4)
    if last:
        t = last["timings"]
        k[0].metric("Retrieval ms", round(t.get("retrieval_ms", 0)))
        k[1].metric("LLM ms", round(t.get("llm_ms", 0)))
        k[2].metric("Expansion ms", round(t.get("expansion_ms", 0)))
        k[3].metric("Total ms", round(t["total_ms"]))
    else:
        st.info("Ask a question in Tab 1 to see latency.")
    bench = CACHE_DIR / "benchmark.json"
    if bench.exists():
        s = json.loads(bench.read_text())["summary"]
        st.subheader("KPIs (last benchmark)")
        m = st.columns(4)
        for col, eng in zip(m, ["Standard RAG", "HyDE RAG", "Graph RAG"]):
            v = s.get(f"{eng}|all", {})
            col.metric(f"{eng}: RCR", f"{v.get('rcr', float('nan')):.2f}")
            col.metric("Faithfulness", f"{v.get('faith', float('nan')):.2f}")
        m[3].metric("Alias resolution", f"{s.get('entity_resolution_accuracy', float('nan')):.2f}")
        m[3].metric("Cross-doc entity linking", f"{s.get('cross_doc_resolution', float('nan')):.2f}")
