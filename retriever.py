"""
retriever.py - Semantic retrieval over the ChromaDB vector store.

Provides:
    classify_query(query)          -> "person" | "place" | "both"
    retrieve(query, query_type, top_k) -> list of result dicts
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    KNOWN_PEOPLE_KEYWORDS,
    KNOWN_PLACES_KEYWORDS,
    TOP_K,
)

# ── Singletons (lazy-loaded) ──────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Query classification ──────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Classify a user query as "person", "place", or "both".

    Logic
    -----
    1. Lower-case the query.
    2. Check whether any known-person keyword appears as a substring.
    3. Check whether any known-place keyword appears as a substring.
    4. Return:
       - "both"   if person AND place keywords found, or neither found
       - "person" if only person keywords found
       - "place"  if only place keywords found
    """
    query_lower = query.lower()

    found_person = any(kw in query_lower for kw in KNOWN_PEOPLE_KEYWORDS)
    found_place = any(kw in query_lower for kw in KNOWN_PLACES_KEYWORDS)

    if found_person and found_place:
        return "both"
    if found_person:
        return "person"
    if found_place:
        return "place"
    return "both"  # default: search everything


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    query_type: str = "both",
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed *query* and retrieve the top-*top_k* relevant chunks from ChromaDB.

    Parameters
    ----------
    query      : natural-language question
    query_type : "person", "place", or "both" — applies a metadata filter
                 when specific (filters to that entity type only)
    top_k      : number of results to return

    Returns
    -------
    List of dicts, each with keys:
        text        - chunk text
        entity_name - name of the Wikipedia article
        type        - "person" or "place"
        chunk_id    - ChromaDB document ID
        distance    - cosine distance (lower = more similar)
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    # Build optional where-filter
    where: dict | None = None
    if query_type in ("person", "place"):
        where = {"type": {"$eq": query_type}}

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Unpack ChromaDB response structure
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output: list[dict] = []
    for doc, meta, dist, chunk_id in zip(docs, metas, dists, ids):
        output.append(
            {
                "text": doc,
                "entity_name": meta.get("entity_name", "Unknown"),
                "type": meta.get("type", "unknown"),
                "chunk_id": chunk_id,
                "distance": round(float(dist), 4),
            }
        )

    return output
