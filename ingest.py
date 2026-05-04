"""
ingest.py - Fetch Wikipedia articles, chunk them, embed, and store in ChromaDB + SQLite.

Usage:
    python ingest.py           # ingest all entities (skips already-ingested ones)
    python ingest.py --reset   # wipe existing data and re-ingest everything
"""

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    PEOPLE,
    PLACES,
    WIKIPEDIA_API_URL,
    WIKIPEDIA_PAGE_URL,
)
from database import delete_all, document_exists, init_db, insert_document

# ── Wikipedia fetching ────────────────────────────────────────────────────────

def fetch_wikipedia_text(title: str) -> tuple[str, str]:
    """
    Fetch plain-text content for *title* via the MediaWiki REST API.

    Returns
    -------
    (text, page_url)
    Raises RuntimeError if the page is missing or the API call fails.
    """
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    api_url = WIKIPEDIA_API_URL.format(title=urllib.parse.quote(title))
    page_url = WIKIPEDIA_PAGE_URL.format(title=encoded_title)

    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "WikipediaRAGAssistant/1.0 (educational project)"},
    )
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 3
                print(f"\n  [RATE-LIMITED] waiting {wait}s ...", end="", flush=True)
                time.sleep(wait)
                last_exc = exc
            else:
                raise RuntimeError(f"Network error fetching '{title}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Network error fetching '{title}': {exc}") from exc
    else:
        raise RuntimeError(f"Network error fetching '{title}': {last_exc}") from last_exc

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise RuntimeError(f"No pages returned for '{title}'")

    page = next(iter(pages.values()))
    if "missing" in page:
        raise RuntimeError(f"Wikipedia page not found for '{title}'")

    raw_text = page.get("extract", "")
    if not raw_text:
        raise RuntimeError(f"Empty extract for '{title}'")

    return raw_text, page_url


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean MediaWiki plain-text output.

    Steps
    -----
    - Normalize unicode to NFC
    - Remove section headers like  == Heading ==  or  === Sub ===
    - Collapse multiple blank lines to a single blank line
    - Strip leading/trailing whitespace from every line
    - Normalise interior whitespace
    """
    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Remove == Section == style headers (any depth)
    text = re.sub(r"={2,}[^=\n]+={2,}", "", text)

    # Remove wiki-style markup remnants ({{...}} and [[...]])
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)

    # Remove HTML entities that may slip through
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Strip lines, collapse blanks
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    # Collapse internal runs of spaces/tabs to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping word-based chunks.

    Parameters
    ----------
    text       : cleaned plain text
    chunk_size : number of words per chunk
    overlap    : number of words shared between consecutive chunks

    Returns
    -------
    List of chunk strings (at least one chunk even for very short texts).
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += chunk_size - overlap  # advance by (chunk_size - overlap)

    return chunks


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    """
    Initialise (or reset) the ChromaDB persistent collection.

    Parameters
    ----------
    reset : if True, delete the existing collection before creating a fresh one.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            print(f"  [ChromaDB] Deleted existing collection '{CHROMA_COLLECTION_NAME}'.")
        except Exception:
            pass  # collection may not exist yet

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Core ingestion logic ──────────────────────────────────────────────────────

def ingest_entity(
    entity_name: str,
    entity_type: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    reset: bool = False,
) -> bool:
    """
    Fetch, chunk, embed and store one entity.

    Returns True on success, False on failure.
    """
    # Skip if already ingested (unless reset mode cleared everything)
    if not reset and document_exists(entity_name):
        print(f"  [SKIP] '{entity_name}' already in DB.")
        return True

    print(f"  [FETCH] {entity_name} ...", end="", flush=True)
    try:
        raw_text, page_url = fetch_wikipedia_text(entity_name)
    except RuntimeError as exc:
        print(f"\n  [ERROR] {exc}")
        return False

    text = clean_text(raw_text)
    chunks = chunk_text(text)
    if not chunks:
        print(f"\n  [ERROR] No text extracted for '{entity_name}'")
        return False

    print(f" {len(chunks)} chunks", end="", flush=True)

    # Embed
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Build ChromaDB documents
    ids = [f"{entity_name.replace(' ', '_')}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "entity_name": entity_name,
            "type": entity_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # Upsert in batches of 100 to stay within ChromaDB limits
    batch_size = 100
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = batch_start + batch_size
        collection.upsert(
            ids=ids[batch_start:batch_end],
            documents=chunks[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )

    # Persist to SQLite
    insert_document(
        entity_name=entity_name,
        entity_type=entity_type,
        url=page_url,
        chunk_count=len(chunks),
        ingested_at=datetime.utcnow().isoformat(),
    )

    print(" ✓")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia pages into the RAG vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing data and re-ingest everything from scratch.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Wikipedia RAG Assistant — Ingestion Pipeline")
    print("=" * 60)

    # Initialise storage
    init_db()
    collection = get_chroma_collection(reset=args.reset)
    if args.reset:
        delete_all()
        print("  [SQLite] Cleared all existing records.")

    # Load embedding model once
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    # Ingest people
    print(f"── Ingesting {len(PEOPLE)} people ──────────────────────────────")
    people_ok = 0
    for name in PEOPLE:
        ok = ingest_entity(name, "person", collection, model, reset=args.reset)
        if ok:
            people_ok += 1
        time.sleep(1.0)  # be polite to Wikipedia's servers

    # Ingest places
    print(f"\n── Ingesting {len(PLACES)} places ──────────────────────────────")
    places_ok = 0
    for name in PLACES:
        ok = ingest_entity(name, "place", collection, model, reset=args.reset)
        if ok:
            places_ok += 1
        time.sleep(0.3)

    # Summary
    total = collection.count()
    print("\n" + "=" * 60)
    print(f"  Ingestion complete.")
    print(f"  People ingested : {people_ok}/{len(PEOPLE)}")
    print(f"  Places ingested : {places_ok}/{len(PLACES)}")
    print(f"  Total chunks in ChromaDB: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
