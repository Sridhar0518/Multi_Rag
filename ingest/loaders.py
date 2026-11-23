"""Document loaders with strict provenance metadata."""
import json
from pathlib import Path

from langchain_core.documents import Document


def _meta(path: Path, **extra):
    return {"source": path.name, "path": str(path), "doc_type": path.suffix.lstrip("."), **extra}


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return [Document(page_content=(p.extract_text() or ""), metadata=_meta(path, page=i + 1))
            for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]


def load_docx(path: Path) -> list[Document]:
    import docx
    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    return [Document(page_content=text, metadata=_meta(path, page=1))]


def load_json(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else [data]
    return [Document(page_content=json.dumps(r, ensure_ascii=False), metadata=_meta(path, page=i + 1))
            for i, r in enumerate(rows)]


def load_text(path: Path) -> list[Document]:
    return [Document(page_content=path.read_text(), metadata=_meta(path, page=1))]


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".json": load_json, ".txt": load_text, ".md": load_text}


def load_file(path: str | Path, practice: str | None = None) -> list[Document]:
    path = Path(path)
    fn = LOADERS.get(path.suffix.lower())
    if fn is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    docs = fn(path)
    for d in docs:
        d.metadata["practice"] = practice or d.metadata.get("practice", "")
    return docs
