# Standard vs HyDE vs Graph RAG — Advisory Benchmark

LangChain + LangGraph implementation of the PwC India Advisory problem statement:
three RAG pipelines compared over Tax, GRC and Deal documents, with a 5-tab Streamlit UI.

> **All documents in `data/sample/` and `test_docs/` are SYNTHETIC.** They imitate the shape of Indian
> regulatory and contract text but are not real law, regulation or contracts. Benchmark numbers come from
> a small corpus (9 documents, 12 questions) and illustrate the method only. The spec's expected rankings
> and latencies (~450 / 1,200 / 850 ms) are hypotheses; this repo measures them rather than hard-coding
> them. With a hosted LLM, latency is dominated by LLM calls (seconds, not milliseconds).

## Run
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# .env: ANTHROPIC_API_KEY=...   (a variable named api_key is also accepted; never commit .env)
streamlit run app.py                                 # UI at http://localhost:8501
python -m eval.benchmark [--no-judge] [--limit N] [--top-k K]   # CLI benchmark -> cache/benchmark.json
python scripts/run_test_docs.py                      # ingest test_docs/ via real loaders, run all 3 engines
pytest                                               # unit tests (LLM mocked)
```
First run downloads `BAAI/bge-large-en-v1.5` (1024-dim) and a small cross-encoder via fastembed/ONNX.
To use OpenAI instead of Anthropic: `pip install langchain-openai` and set `LLM_PROVIDER=openai`.

## Try it with the test documents
`test_docs/` holds a PDF, DOCX and JSON plus `TEST_QUESTIONS.md` (9 questions with expected facts,
including multi-hop and vocabulary-mismatch cases). In Tab 1: choose the practice area, upload the
matching files (Tax files under *Tax*, the SOP under *GRC*, the SPA under *Deal*), click
**Ingest uploaded files**, pick an engine and ask the questions. Regenerate the files with
`python scripts/make_test_docs.py test_docs`.

## Architecture
| Spec step | Implementation |
|---|---|
| 1 Ingestion | `ingest/loaders.py` (PDF via pypdf, DOCX, JSON, TXT) with provenance metadata |
| 2 Chunking + entities | `ingest/chunking.py` (600/60, clause-aware separators), `ingest/entities.py` (LLM extraction, cached in `cache/`) |
| 3 Normalisation | alias table in `ingest/entities.py` (CBDT / Income Tax Dept / Tax Authority → CBDT) |
| 4 Embeddings | bge-large-en-v1.5 text vectors; Node2Vec-style structural embeddings (random walks + SVD) in `NetworkXGraph.node_embeddings` |
| 5 Dual index | Chroma HNSW (cosine) + BM25, and NetworkX adjacency index |
| 6 Orchestration | `rag/workflow.py` — LangGraph: Query Expansion → Hybrid Search → RRF → Rerank → Synthesis |

Engines differ only in expansion/search: `rag/standard.py` (dense), `rag/hyde.py` (N passages),
`rag/graph_rag.py` (entity linking + structurally similar entity + k-hop + dense + BM25).
Reranking uses a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`), falling back to lexical overlap.

## KPIs (`eval/`)
- **Retrieval Context Recall:** share of ground-truth facts present in the retrieved chunks.
- **Faithfulness:** LLM-judge ratio of supported claims, judged against exactly what the generator saw
  (chunks plus, for Graph RAG, the graph relations).
- **Entity resolution:** alias-table accuracy, plus cross-document linking (does one graph node link
  chunks from all expected source documents).
- **Latency:** expansion / retrieval / LLM / total, per query.

## Known deviations from the spec
- **Backends:** ChromaDB + NetworkX (no Docker/Neo4j here). `GraphStore` is an interface for a Neo4j swap.
- **Tokens:** chunk sizes are counted with a regex word/punctuation tokenizer, an approximation of model tokens.
- **Node2Vec:** random-walk co-occurrence + SVD rather than the `node2vec` package. It is used in retrieval
  only to add the single most structurally similar entity to the query's seed entities.
- **Tab 2:** node and relation selection use multiselects beside the 3D Plotly canvas (Plotly 3D has no
  click-select callback in Streamlit).
- **Tab 3:** nodes can be disabled (pass-through) and individual edges can be cut (the downstream node
  receives nothing from the upstream node). The DAG itself is linear.
- **Unstructured:** `UnstructuredPDFLoader` is replaced by pypdf to avoid heavy system dependencies.
- **Temperature:** `claude-sonnet-5-5` rejects non-default temperature, so none is set.
- **Small corpus:** each sample document is a single chunk, so recall saturates unless `--top-k` is low;
  results will not discriminate between engines the way a large real corpus would.
