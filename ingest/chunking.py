"""Clause-aware chunking: RecursiveCharacterTextSplitter sized in (approximate) tokens."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TOK = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOK.findall(text))


def chunk_documents(docs: list[Document], chunk_size=600, chunk_overlap=60) -> list[Document]:
    # Separators prefer clause/section boundaries so legal clauses are not split mid-way.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=count_tokens,
        separators=["\n\n", "\nSection ", "\nClause ", "\nRule ", "\n", ". ", " ", ""],
    )
    out = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            md = dict(d.metadata)
            md["chunk_id"] = f"{md['source']}::p{md.get('page', 1)}::c{i}"
            out.append(Document(page_content=piece, metadata=md))
    return out
