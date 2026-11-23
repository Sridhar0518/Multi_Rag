"""Multi-hypothesis HyDE: N synthetic passages from distinct regulatory angles."""
from config import text_of
from prompts import HYDE_ANGLES

PROMPT = """Write a concise (<=120 words) hypothetical passage, in the style of an Indian regulatory
document or contract, that would answer the question from this angle: {angle}.
Do not add disclaimers. Question: {q}"""


def expand(ctx, state):
    q = state["question"]
    passages = []
    for angle in HYDE_ANGLES[: ctx.hp.hyde_n]:
        passages.append(text_of(ctx.llm.invoke(PROMPT.format(angle=angle, q=q))).strip())
    return {"queries": passages, "seeds": [], "hyde_passages": passages}


def search(ctx, state):
    k = ctx.hp.top_k * 2
    lists = [ctx.index.dense(p, k, state.get("practice_filter")) for p in state["queries"]]
    return {"rank_lists": lists, "graph_edges": []}
