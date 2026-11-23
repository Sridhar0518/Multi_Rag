"""Graph RAG: entity linking + k-hop expansion fused with dense and BM25 retrieval."""

def expand(ctx, state):
    seeds = ctx.graph.link_query(state["question"])
    if seeds and hasattr(ctx.graph, "similar_nodes"):
        # Node2Vec-style structural embeddings: add the most structurally similar entities
        seeds = seeds + ctx.graph.similar_nodes(seeds, top_n=1)
    return {"queries": [state["question"]], "seeds": seeds}


def search(ctx, state):
    q, k = state["question"], ctx.hp.top_k * 2
    pf = state.get("practice_filter")
    lists = [ctx.index.dense(q, k, pf), ctx.index.sparse(q, k, pf)]
    edges = []
    if state.get("seeds"):
        nodes, edges = ctx.graph.k_hop(state["seeds"], ctx.hp.hop_k)
        # rank graph chunks: seed-adjacent first (edge evidence), then node-linked chunks
        seen, glist = set(), []
        for _, _, _, cid in edges:
            if cid not in seen:
                seen.add(cid); glist.append(cid)
        for cid in ctx.graph.chunks_for(nodes):
            if cid not in seen:
                seen.add(cid); glist.append(cid)
        lists.append([c for c in glist if c in ctx.index.docs][:k])
    return {"rank_lists": lists, "graph_edges": edges}
