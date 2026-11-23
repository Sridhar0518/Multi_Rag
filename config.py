"""Central configuration: hyperparameters and LLM/provider loading.

API key values are never printed or logged.
"""
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # 1024-dim
LLM_MODEL_ANTHROPIC = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
CACHE_DIR = ROOT / "cache"
SAMPLE_DIR = ROOT / "data" / "sample"


@dataclass
class Hyperparams:
    chunk_size: int = 600          # characters-approx tokens (see chunking.py)
    chunk_overlap: int = 60
    hyde_n: int = 3                # Multi-HyDE passage count
    hop_k: int = 2                 # graph traversal radius
    rrf_alpha: int = 60            # RRF constant
    top_k: int = 3

    def as_dict(self):
        return asdict(self)


def text_of(resp) -> str:
    """Normalise a chat response to plain text (content may be a str or a list of blocks)."""
    c = getattr(resp, "content", resp)
    if isinstance(c, str):
        return c
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in c
                   if isinstance(b, str) or b.get("type") == "text")


def get_api_key() -> str | None:
    for name in ("ANTHROPIC_API_KEY", "api_key", "API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    return None


def get_llm(temperature: float | None = None, max_tokens: int = 1500):
    key = get_api_key()
    if not key:
        raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY (or api_key) in .env")
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kw = {} if temperature is None else {"temperature": temperature}
        return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o"), api_key=key, max_tokens=max_tokens, **kw)
    from langchain_anthropic import ChatAnthropic
    # claude-sonnet-5-5 rejects non-default temperature, so it is only passed when explicitly set.
    kw = {} if temperature is None else {"temperature": temperature}
    return ChatAnthropic(model=LLM_MODEL_ANTHROPIC, api_key=key, max_tokens=max_tokens, **kw)
