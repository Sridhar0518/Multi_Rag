"""Standard vector RAG: no expansion, dense top-k only."""

def expand(ctx, state):
    return {"queries": [state["question"]], "seeds": []}

def search(ctx, state):
    k = ctx.hp.top_k * 2
    return {"rank_lists": [ctx.index.dense(state["question"], k, state.get("practice_filter"))], "graph_edges": []}
