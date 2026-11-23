"""LLM-guided entity-relation extraction + alias normalisation, with on-disk cache."""
import hashlib
import json
import re

from langchain_core.documents import Document

from config import CACHE_DIR, get_llm, text_of

# Canonical entity -> aliases (case-insensitive). Extended automatically by the LLM output.
ALIASES = {
    "CBDT": ["cbdt", "central board of direct taxes", "income tax dept", "income tax department", "tax authority", "it department"],
    "CBIC": ["cbic", "central board of indirect taxes and customs", "customs and gst board"],
    "RBI": ["rbi", "reserve bank of india", "central bank"],
    "SEBI": ["sebi", "securities and exchange board of india"],
    "GST Council": ["gst council", "goods and services tax council"],
    "PMLA": ["pmla", "prevention of money laundering act"],
    "KYC": ["kyc", "know your customer"],
}
_ALIAS_LOOKUP = {a: c for c, al in ALIASES.items() for a in al}

EXTRACT_PROMPT = """Extract compliance entities and relations from the text. Return ONLY JSON:
{{"entities":[{{"name":"...","type":"Regulator|Statute|Section|Obligation|Penalty|Party|Clause|Threshold|Concept"}}],
"relations":[{{"source":"...","relation":"...","target":"..."}}]}}
Keep names short and exact. Text:
{text}"""


def normalize(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().lower())
    return _ALIAS_LOOKUP.get(key, name.strip())


def _parse_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(m.group(0)) if m else {"entities": [], "relations": []}
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def extract_chunk(chunk: Document, llm=None) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:20]
    f = CACHE_DIR / f"ent_{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    llm = llm or get_llm(max_tokens=1200)
    res = _parse_json(text_of(llm.invoke(EXTRACT_PROMPT.format(text=chunk.page_content))))
    f.write_text(json.dumps(res))
    return res


def extract_all(chunks: list[Document], llm=None) -> list[dict]:
    """Return per-chunk extraction with normalised names."""
    out = []
    for c in chunks:
        r = extract_chunk(c, llm)
        ents = [{"name": normalize(e["name"]), "type": e.get("type", "Concept")}
                for e in r.get("entities", []) if e.get("name")]
        rels = [{"source": normalize(x["source"]), "relation": x.get("relation", "related_to"),
                 "target": normalize(x["target"])}
                for x in r.get("relations", []) if x.get("source") and x.get("target")]
        out.append({"chunk_id": c.metadata["chunk_id"], "entities": ents, "relations": rels})
    return out
