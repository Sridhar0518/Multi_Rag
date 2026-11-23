"""End-to-end ingestion: load -> chunk -> entity extraction -> dual index."""
from pathlib import Path

from config import Hyperparams
from index.graph_store import NetworkXGraph
from index.vector_store import VectorIndex
from ingest.chunking import chunk_documents
from ingest.entities import extract_all
from ingest.loaders import load_file


def ingest_files(paths: list, practice_of, index: VectorIndex, graph: NetworkXGraph,
                 hp: Hyperparams, llm=None, extract=True):
    """practice_of: callable(Path)->practice name."""
    all_chunks = []
    for p in map(Path, paths):
        docs = load_file(p, practice_of(p))
        all_chunks += chunk_documents(docs, hp.chunk_size, hp.chunk_overlap)
    index.add(all_chunks)
    if extract:
        for ex in extract_all(all_chunks, llm):
            graph.add_extraction(ex)
    return all_chunks
