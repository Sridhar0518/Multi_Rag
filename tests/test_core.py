import networkx as nx
from langchain_core.documents import Document

from config import Hyperparams
from eval.metrics import entity_resolution_accuracy, retrieval_context_recall
from index.embeddings import HashEmbedder
from index.graph_store import NetworkXGraph
from index.vector_store import VectorIndex
from ingest.chunking import chunk_documents, count_tokens
from ingest.entities import normalize
from prompts import PRACTICE_AREAS, system_prompt
from rag.common import RagContext
from rag.fusion import rrf
from rag.workflow import NODES, run


class FakeLLM:
    def invoke(self, p):
        class R: content = "answer [S1]"
        return R()


def mk_ctx():
    idx, g = VectorIndex(HashEmbedder()), NetworkXGraph()
    docs = [Document(page_content=t, metadata={"chunk_id": f"d::c{i}", "source": "d", "practice": "P"})
            for i, t in enumerate(["CBDT imposes penalty Section 271C", "Section 271C relates to TDS", "unrelated text on ISD"])]
    idx.add(docs)
    g.add_extraction({"chunk_id": "d::c0", "entities": [{"name": "CBDT", "type": "Regulator"}],
                      "relations": [{"source": "CBDT", "relation": "imposes", "target": "Section 271C"}]})
    g.add_extraction({"chunk_id": "d::c1", "entities": [], "relations": [{"source": "Section 271C", "relation": "relates_to", "target": "TDS"}]})
    return RagContext(idx, g, FakeLLM(), Hyperparams())


def test_chunk_size_and_overlap():
    text = "\n\n".join(f"Section {i}. " + "word " * 100 for i in range(20))
    ch = chunk_documents([Document(page_content=text, metadata={"source": "x", "page": 1})], 600, 60)
    assert len(ch) > 1 and all(count_tokens(c.page_content) <= 600 for c in ch)


def test_alias_normalisation():
    assert normalize("Income Tax Dept") == "CBDT" and normalize("tax authority") == "CBDT"
    assert entity_resolution_accuracy([["Income Tax Dept", "CBDT"], ["Foo", "Bar"]]) == 0.5


def test_k_hop():
    g = mk_ctx().graph
    n1, _ = g.k_hop(["CBDT"], 1)
    n2, _ = g.k_hop(["CBDT"], 2)
    assert "TDS" not in n1 and "TDS" in n2


def test_link_query_alias():
    assert "CBDT" in mk_ctx().graph.link_query("what does the Income Tax Dept do?")


def test_rrf_math():
    r = dict(rrf([["a", "b"], ["b", "a"]], 60))
    assert abs(r["a"] - (1 / 61 + 1 / 62)) < 1e-9


def test_node_toggle_passthrough():
    ctx = mk_ctx()
    out = run(ctx, "penalty Section 271C", "Standard RAG", "Tax & Regulatory Services", disabled=["synthesis"])
    assert out["answer"].startswith("(synthesis disabled)")
    out = run(ctx, "penalty", "Standard RAG", "Tax & Regulatory Services", disabled=["hybrid_search"])
    assert out["final_ids"] == []


def test_graph_rag_pulls_multihop_chunk():
    ctx = mk_ctx()
    out = run(ctx, "What does CBDT do?", "Graph RAG", "Tax & Regulatory Services")
    assert "d::c1" in out["final_ids"] and out["graph_edges"]


def test_guardrails_per_practice():
    for p, v in PRACTICE_AREAS.items():
        assert v["guardrail"] in system_prompt(p)
    assert len(NODES) == 5


def test_rcr():
    assert retrieval_context_recall(["271C", "zzz"], [Document(page_content="Section 271C")]) == 0.5


def test_edge_cut_drops_data():
    ctx = mk_ctx()
    out = run(ctx, "penalty Section 271C", "Standard RAG", "Tax & Regulatory Services",
              disabled_edges=["hybrid_search->rrf_fusion"])
    assert out["final_ids"] == [] and out["chunks"] == []


def test_similar_nodes_uses_structure():
    g = mk_ctx().graph
    assert isinstance(g.similar_nodes(["CBDT"], top_n=1), list)


def test_cross_doc_resolution():
    from eval.metrics import cross_doc_resolution
    g = NetworkXGraph()
    g.add_extraction({"chunk_id": "a.txt::c0", "entities": [{"name": "Section 1", "type": "Section"}], "relations": []})
    g.add_extraction({"chunk_id": "b.txt::c0", "entities": [{"name": "Section 1", "type": "Section"}], "relations": []})
    assert cross_doc_resolution(g, [{"entity": "Section 1", "sources": ["a.txt", "b.txt"]},
                                    {"entity": "Missing", "sources": ["a.txt"]}]) == 0.5


def test_real_file_loaders(tmp_path):
    import subprocess, sys
    from pathlib import Path
    from ingest.loaders import load_file
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "scripts" / "make_test_docs.py"), str(tmp_path)], check=True)
    for name, needle in [("Tax_Circular_TDS_Freelancers.pdf", "194Z"), ("GRC_Vendor_Risk_SOP.docx", "V-04"), ("Tax_Rates_Table.json", "271Z")]:
        docs = load_file(tmp_path / name, "P")
        assert needle in " ".join(d.page_content for d in docs) and docs[0].metadata["source"] == name


def test_ingest_pdf_docx_without_llm(tmp_path):
    import subprocess, sys
    from pathlib import Path
    from ingest.pipeline import ingest_files
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "scripts" / "make_test_docs.py"), str(tmp_path)], check=True)
    idx, g = VectorIndex(HashEmbedder()), NetworkXGraph()
    ch = ingest_files([tmp_path / "Deal_SPA_Project_Meridian.pdf", tmp_path / "GRC_Vendor_Risk_SOP.docx"], lambda p: "P", idx, g, Hyperparams(), extract=False)
    assert ch and idx.dense("indemnity cap", 1)[0].startswith("Deal_SPA")
