"""Run all three engines over the QA set and aggregate KPIs.

Usage: python -m eval.benchmark [--no-judge] [--limit N]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from config import CACHE_DIR, SAMPLE_DIR, ROOT, Hyperparams, get_llm
from eval.metrics import cross_doc_resolution, entity_resolution_accuracy, faithfulness, graph_facts, retrieval_context_recall
from index.embeddings import BgeEmbedder
from index.graph_store import NetworkXGraph
from index.vector_store import VectorIndex
from ingest.pipeline import ingest_files
from prompts import PRACTICE_AREAS
from rag.common import RagContext
from rag.workflow import ENGINES, run

PREFIX = {"tax": "Tax & Regulatory Services", "grc": "Risk Advisory & Governance (GRC)",
          "deal": "Deal Advisory & Corporate Finance"}


def practice_of(p: Path) -> str:
    return PREFIX[p.name.split("_")[0]]


def build_context(hp: Hyperparams | None = None, embedder=None, llm=None) -> RagContext:
    hp = hp or Hyperparams()
    llm = llm or get_llm()
    idx, g = VectorIndex(embedder or BgeEmbedder()), NetworkXGraph()
    ingest_files(sorted(SAMPLE_DIR.glob("*.txt")), practice_of, idx, g, hp, llm)
    return RagContext(idx, g, llm, hp)


def run_benchmark(ctx: RagContext, qa: list[dict], judge=True, engines=None, progress=None):
    rows = []
    for i, q in enumerate(qa):
        for eng in engines or ENGINES:
            out = run(ctx, q["question"], eng, q["practice"])
            rows.append({
                "id": q["id"], "type": q["type"], "engine": eng,
                "rcr": retrieval_context_recall(q["facts"], out["chunks"]),
                "faith": faithfulness(out["answer"], out["chunks"], ctx.llm, graph_facts(out.get("graph_edges", [])) if eng == "Graph RAG" else "") if judge else float("nan"),
                **{k: round(v, 1) for k, v in out["timings"].items()},
                "answer": out["answer"],
            })
            if progress:
                progress((i * len(engines or ENGINES)) / (len(qa) * len(engines or ENGINES)), q["id"], eng)
    return rows


def summarize(rows: list[dict]) -> dict:
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for grp in ("all", r["type"]):
            agg[(r["engine"], grp)]["rcr"].append(r["rcr"])
            if not math.isnan(r["faith"]):
                agg[(r["engine"], grp)]["faith"].append(r["faith"])
            agg[(r["engine"], grp)]["total_ms"].append(r["total_ms"])
    return {f"{e}|{g}": {m: sum(v) / len(v) for m, v in d.items() if v} for (e, g), d in agg.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--top-k", type=int, default=None, help="final context size (use 2 to stress retrieval)")
    a = ap.parse_args()
    qa = json.loads((ROOT / "data" / "qa.json").read_text())[: a.limit]
    ctx = build_context(Hyperparams(**({"top_k": a.top_k} if a.top_k else {})))
    rows = run_benchmark(ctx, qa, judge=not a.no_judge)
    gold = json.loads((ROOT / "data" / "entity_gold.json").read_text())
    summ = summarize(rows)
    summ["entity_resolution_accuracy"] = entity_resolution_accuracy(gold["alias_pairs"])
    summ["cross_doc_resolution"] = cross_doc_resolution(ctx.graph, gold["cross_doc"])
    (CACHE_DIR / "benchmark.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    for k, v in summ.items():
        print(k, v)
