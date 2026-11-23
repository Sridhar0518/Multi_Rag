"""Ingest test_docs/ through the REAL loaders + embedder + LLM, run all 3 engines on test_questions.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import Hyperparams, get_llm  # noqa: E402
from eval.metrics import retrieval_context_recall  # noqa: E402
from index.embeddings import BgeEmbedder  # noqa: E402
from index.graph_store import NetworkXGraph  # noqa: E402
from index.vector_store import VectorIndex  # noqa: E402
from ingest.pipeline import ingest_files  # noqa: E402
from rag.common import RagContext  # noqa: E402
from rag.workflow import ENGINES, run  # noqa: E402

D = ROOT / "test_docs"
PRAC = {"Tax": "Tax & Regulatory Services", "GRC": "Risk Advisory & Governance (GRC)", "Deal": "Deal Advisory & Corporate Finance"}
ctx = RagContext(VectorIndex(BgeEmbedder()), NetworkXGraph(), get_llm(), Hyperparams())
files = [p for p in sorted(D.iterdir()) if p.suffix in (".pdf", ".docx", ".json") and p.name != "test_questions.json"]
ch = ingest_files(files, lambda p: PRAC[p.name.split("_")[0]], ctx.index, ctx.graph, ctx.hp, ctx.llm)
print(f"ingested {len(files)} files -> {len(ch)} chunks, {ctx.graph.g.number_of_nodes()} entities, {ctx.graph.g.number_of_edges()} relations")
res = {}
for q in json.loads((D / "test_questions.json").read_text()):
    for e in ENGINES:
        o = run(ctx, q["question"], e, q["practice"])
        r = retrieval_context_recall(q["facts"], o["chunks"])
        res.setdefault(e, []).append(r)
        hit = sum(f.lower() in o["answer"].lower() for f in q["facts"])
        print(f"{q['id']} {e:13s} RCR={r:.2f} answer_has={hit}/{len(q['facts'])} total={o['timings']['total_ms']:.0f}ms  [{q['type']}]")
for e, v in res.items():
    print(f"MEAN RCR {e}: {sum(v)/len(v):.2f}")
