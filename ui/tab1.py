import tempfile
from pathlib import Path

import streamlit as st

from config import SAMPLE_DIR
from eval.benchmark import practice_of as sample_practice
from ingest.pipeline import ingest_files
from prompts import PRACTICE_AREAS
from rag.workflow import ENGINES, run
from ui.state import get_ctx


def render():
    ctx = get_ctx()
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Data ingestion")
        practice = st.selectbox("Practice area", list(PRACTICE_AREAS), key="practice")
        files = st.file_uploader("Upload PDF / DOCX / JSON / TXT", type=["pdf", "docx", "json", "txt"],
                                 accept_multiple_files=True)
        if st.button("Ingest uploaded files", disabled=not files):
            tmp = Path(tempfile.mkdtemp())
            paths = []
            for f in files:
                p = tmp / f.name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            with st.spinner("Chunking, extracting entities, indexing..."):
                ch = ingest_files(paths, lambda p: practice, ctx.index, ctx.graph, ctx.hp, ctx.llm)
            st.success(f"Indexed {len(ch)} chunks")
        if st.button("Load synthetic sample corpus"):
            with st.spinner("Ingesting sample corpus (entity extraction is cached)..."):
                ch = ingest_files(sorted(SAMPLE_DIR.glob("*.txt")), sample_practice, ctx.index, ctx.graph, ctx.hp, ctx.llm)
            st.success(f"Indexed {len(ch)} chunks from the synthetic sample corpus")
        st.caption(f"{len(ctx.index.docs)} chunks | {ctx.graph.g.number_of_nodes()} entities | {ctx.graph.g.number_of_edges()} relations")
        engine = st.radio("RAG engine", list(ENGINES), key="engine")
        only_practice = st.checkbox("Restrict retrieval to selected practice area", value=True)
    with right:
        st.subheader("Chat")
        for m in st.session_state.history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    with st.expander("Sources & timings"):
                        for i, c in enumerate(m["sources"], 1):
                            st.markdown(f"**[S{i}]** `{c.metadata['source']}` — {c.page_content[:300]}...")
                        st.json({k: round(v) for k, v in m["timings"].items()})
        q = st.chat_input("Ask a question about the indexed documents")
        if q:
            if not ctx.index.docs:
                st.warning("Ingest documents first.")
                return
            st.session_state.history.append({"role": "user", "content": q})
            with st.spinner(f"Running {engine}..."):
                out = run(ctx, q, engine, practice, st.session_state.disabled_nodes,
                          practice if only_practice else None, st.session_state.disabled_edges)
            st.session_state.last = out
            st.session_state.history.append({"role": "assistant", "content": out["answer"],
                                             "sources": out["chunks"], "timings": out["timings"]})
            st.rerun()
