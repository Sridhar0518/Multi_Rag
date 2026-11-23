import streamlit as st

from rag.workflow import NODES
from ui.state import get_ctx

LABELS = {"query_expansion": "Query Expansion", "hybrid_search": "Hybrid Vector/Graph Search",
          "rrf_fusion": "RRF Fusion", "rerank": "Rerank", "synthesis": "Synthesis"}


def render():
    get_ctx()
    st.subheader("LangGraph workflow canvas")
    cols = st.columns(len(NODES))
    disabled = []
    for c, n in zip(cols, NODES):
        if not c.checkbox(LABELS[n], value=n not in st.session_state.disabled_nodes, key=f"node_{n}"):
            disabled.append(n)
    st.session_state.disabled_nodes = disabled
    edge_names = [f"{a}->{b}" for a, b in zip(NODES, NODES[1:])]
    cut = st.multiselect("Disable edges (the downstream node receives nothing from the upstream node)", edge_names,
                         default=st.session_state.disabled_edges,
                         format_func=lambda e: " → ".join(LABELS[x] for x in e.split("->")))
    st.session_state.disabled_edges = cut
    dot = ["digraph G { rankdir=LR; node [shape=box, style=filled, fontname=Helvetica];", 'START [shape=circle, fillcolor="#eeeeee"]; END [shape=circle, fillcolor="#eeeeee"];']
    for n in NODES:
        dot.append(f'{n} [label="{LABELS[n]}", fillcolor="{"#f4c7a8" if n in disabled else "#d04a02"}", fontcolor="{"#888888" if n in disabled else "white"}"];')
    chain = ["START"] + NODES + ["END"]
    for a, b in zip(chain, chain[1:]):
        dead = f"{a}->{b}" in cut
        dot.append(f'{a} -> {b} [style={"dashed" if dead else "solid"}, color="{"#cc0000" if dead else "black"}"{", label=cut" if dead else ""}];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot))
    st.caption("Disabled nodes become pass-throughs (e.g. no RRF = first rank list only; no synthesis = raw retrieved IDs). "
               "Settings apply to the next question in Tab 1.")
