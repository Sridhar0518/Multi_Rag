"""KPIs: RCR, faithfulness, entity-resolution accuracy, latency."""
import json
import re

from config import text_of
from ingest.entities import normalize


def retrieval_context_recall(facts: list[str], chunks) -> float:
    if not facts:
        return 1.0
    text = " ".join(c.page_content.lower() for c in chunks)
    return sum(f.lower() in text for f in facts) / len(facts)


JUDGE = """You are an auditor. Split the ANSWER into atomic factual claims, then count how many are
directly supported by the SOURCES. Ignore formatting, "Gaps" disclosures and generic advice.
Return ONLY JSON: {{"total": <int>, "supported": <int>}}
SOURCES:
{sources}
ANSWER:
{answer}"""


def graph_facts(edges) -> str:
    return "\n".join(f"{u} --{r}--> {v}" for u, v, r, _ in edges[:25])


def faithfulness(answer: str, chunks, llm, extra_facts: str = "") -> float:
    """Judge sees exactly what the generator saw: chunks plus (for Graph RAG) graph relations."""
    src = "\n\n".join(c.page_content for c in chunks) + ("\n\n[Graph relations]\n" + extra_facts if extra_facts else "")
    raw = text_of(llm.invoke(JUDGE.format(sources=src, answer=answer)))
    m = re.search(r"\{.*?\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        return min(1.0, d["supported"] / d["total"]) if d["total"] else 1.0
    except Exception:
        return float("nan")


def cross_doc_resolution(graph, gold: list[dict]) -> float:
    """Fraction of gold entities whose single graph node links chunks from ALL expected source docs."""
    if not gold:
        return 0.0
    ok = 0
    for g in gold:
        node = normalize(g["entity"])
        # tolerate extractor naming variants: any node whose name contains the gold string
        cands = [n for n in graph.g if n == node or g["entity"].lower() in n.lower()]
        for n in cands:
            srcs = {c.split("::")[0] for c in graph.g.nodes[n]["chunks"]}
            if set(g["sources"]) <= srcs:
                ok += 1
                break
    return ok / len(gold)


def entity_resolution_accuracy(alias_pairs: list[list[str]]) -> float:
    ok = sum(normalize(a) == b for a, b in alias_pairs)
    return ok / len(alias_pairs) if alias_pairs else 0.0
