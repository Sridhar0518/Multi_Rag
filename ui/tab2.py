import math

import plotly.graph_objects as go
import streamlit as st

from ui.state import get_ctx


def _sphere(n):
    """Fibonacci sphere layout."""
    pts = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0, 1 - y * y))
        th = math.pi * (3 - math.sqrt(5)) * i
        pts.append((r * math.cos(th), y, r * math.sin(th)))
    return pts


def render():
    ctx = get_ctx()
    g = ctx.graph.g
    st.subheader("Spherical knowledge graph")
    if g.number_of_nodes() == 0:
        st.info("No graph yet. Ingest documents in Tab 1.")
        return
    nodes = sorted(g.nodes)
    pos = dict(zip(nodes, _sphere(len(nodes))))
    c1, c2 = st.columns([3, 1])
    with c2:
        active = st.multiselect("Active nodes (unselect to exclude from retrieval)", nodes,
                                default=[n for n in nodes if n not in ctx.graph.disabled_nodes])
        ctx.graph.disabled_nodes = set(nodes) - set(active)
        st.caption("Excluded nodes are ignored by Graph RAG entity linking and k-hop traversal.")
    with c1:
        ex, ey, ez = [], [], []
        for u, v in {(u, v) for u, v in g.edges()}:
            if u in ctx.graph.disabled_nodes or v in ctx.graph.disabled_nodes:
                continue
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]; ez += [pos[u][2], pos[v][2], None]
        act = [n for n in nodes if n in active]
        off = [n for n in nodes if n not in active]
        fig = go.Figure([
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#9aa5b1", width=2), hoverinfo="none", name="edges"),
            go.Scatter3d(x=[pos[n][0] for n in act], y=[pos[n][1] for n in act], z=[pos[n][2] for n in act],
                         mode="markers+text", text=act, textposition="top center", name="active",
                         marker=dict(size=6, color="#d04a02"), hovertext=[g.nodes[n]["type"] for n in act]),
            go.Scatter3d(x=[pos[n][0] for n in off], y=[pos[n][1] for n in off], z=[pos[n][2] for n in off],
                         mode="markers", name="excluded", marker=dict(size=4, color="#cccccc"), text=off),
        ])
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=0, b=0), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
        st.plotly_chart(fig, width="stretch")
    all_edges = sorted({f"{u} --{d['relation']}--> {v}" for u, v, d in g.edges(data=True)})
    keys = {f"{u} --{d['relation']}--> {v}": (u, v, d["relation"]) for u, v, d in g.edges(data=True)}
    off_e = st.multiselect("Excluded relations (ignored by k-hop traversal)", all_edges,
                           default=[e for e in all_edges if keys[e] in ctx.graph.disabled_edges])
    ctx.graph.disabled_edges = {keys[e] for e in off_e}
    with st.expander("Edges"):
        st.dataframe([{"source": u, "relation": d["relation"], "target": v, "chunk": d["chunk_id"]} for u, v, d in g.edges(data=True)])
