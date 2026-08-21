################################################################################
# Karma Edge - tools/retrieval.py
#
# Free retrieval only. Two backends, no paid vector DB:
#   bm25   (default) : rank_bm25, pure python, zero install pain. Instant.
#   chroma           : local persistent Chroma + free embeddings.
#
# Corpus = policy docs, vendor terms, merchandising playbooks in knowledge/.
# This is what lets an agent say "this violates the markdown policy, section 3"
# instead of "discounting seems high".
################################################################################
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import ROOT, settings

KNOWLEDGE_DIR = ROOT / "knowledge"
_CHUNK, _OVERLAP = 900, 150
_cache: dict = {}


def _chunks() -> List[Tuple[str, str]]:
    if "chunks" in _cache:
        return _cache["chunks"]
    out: List[Tuple[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("**/*.md")):
        text = path.read_text(errors="ignore")
        start = 0
        while start < len(text):
            out.append((path.name, text[start : start + _CHUNK]))
            start += _CHUNK - _OVERLAP
    _cache["chunks"] = out
    return out


def _bm25_search(query: str, k: int) -> List[Tuple[float, str, str]]:
    from rank_bm25 import BM25Okapi

    docs = _chunks()
    if not docs:
        return []
    if "bm25" not in _cache:
        _cache["bm25"] = BM25Okapi([d[1].lower().split() for d in docs])
    scores = _cache["bm25"].get_scores(query.lower().split())
    ranked = sorted(zip(scores, docs), key=lambda x: -x[0])[:k]
    return [(float(s), d[0], d[1]) for s, d in ranked if s > 0]


def _chroma_search(query: str, k: int) -> List[Tuple[float, str, str]]:
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    ef = embedding_functions.DefaultEmbeddingFunction()  # local MiniLM, free
    col = client.get_or_create_collection("karma_knowledge", embedding_function=ef)
    if col.count() == 0:
        docs = _chunks()
        col.add(
            ids=[f"c{i}" for i in range(len(docs))],
            documents=[d[1] for d in docs],
            metadatas=[{"source": d[0]} for d in docs],
        )
    res = col.query(query_texts=[query], n_results=k)
    return [
        (1.0 - float(dist), meta.get("source", "?"), doc)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


class KBInput(BaseModel):
    query: str = Field(description="What policy or playbook rule you need to check.")
    k: int = Field(default=4, description="Number of passages to return.")


@tool("search_policy", args_schema=KBInput)
def search_policy(query: str, k: int = 4) -> str:
    """Search internal policy documents, vendor terms and merchandising playbooks
    (markdown, retrieval, BM25 by default, local Chroma optional). Cite the source
    filename in your finding so the owner can argue with the document, not you."""
    mode = settings.retrieval_mode
    try:
        hits = _chroma_search(query, k) if mode == "chroma" else _bm25_search(query, k)
    except Exception as exc:
        try:
            hits = _bm25_search(query, k)
        except Exception:
            return f"RETRIEVAL UNAVAILABLE ({type(exc).__name__}: {exc}). pip install rank-bm25"
    if not hits:
        return "No policy passage matched. Do not invent a policy: say the rule is undocumented."
    return "\n\n".join(f"[{src} score={score:.2f}]\n{text.strip()[:700]}" for score, src, text in hits)
