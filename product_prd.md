# Product Requirements Document
## Wikipedia RAG Assistant

**Version:** 1.0  
**Date:** 2026-05-04  
**Status:** Draft

---

## 1. Problem Statement

Students, researchers, and curious learners frequently want concise, accurate answers about well-known historical and contemporary figures, as well as famous landmarks and natural wonders. General web search returns scattered, ad-heavy pages; LLMs hallucinate facts; and Wikipedia itself requires users to read entire articles to find a single answer.

This project delivers a **retrieval-augmented generation (RAG)** assistant that:
1. Grounds every answer in verified Wikipedia text (no hallucination from training data).
2. Runs entirely on the user's local machine (no external API costs, no data leaving the device).
3. Provides a conversational chat interface familiar to users of ChatGPT / Claude.

---

## 2. User Personas

### Persona A — The Student
- **Who:** Undergraduate student writing an essay or preparing for an exam.
- **Goal:** Get accurate, sourced facts about a historical figure or landmark quickly.
- **Pain point:** Wikipedia articles are long; LLMs make up citations.
- **How this helps:** Conversational Q&A with source attribution and no fabricated facts.

### Persona B — The Curious Explorer
- **Who:** General adult user who enjoys learning about history and travel.
- **Goal:** Ask natural-language questions ("Why is the Colosseum famous?") and get concise answers.
- **Pain point:** Search engines return too many links; they want a direct answer.
- **How this helps:** Single-turn or multi-turn chat with focused, context-grounded responses.

### Persona C — The Developer / Researcher
- **Who:** ML engineer or NLP researcher evaluating local RAG architectures.
- **Goal:** Understand how sentence-transformers + ChromaDB + a local LLM perform end-to-end.
- **Pain point:** Most RAG demos depend on paid cloud APIs.
- **How this helps:** A fully self-contained, reproducible local pipeline with transparent components.

---

## 3. Core Features

### F1 — Wikipedia Ingestion
- Fetch plain-text article content using the MediaWiki REST API (`urllib`, no third-party wiki library).
- Clean markup artefacts (section headers, template tags, HTML entities).
- Support incremental ingestion (skip already-ingested entities) and full reset (`--reset` flag).
- Ingest 20 people and 20 places (40 articles total).

### F2 — Text Chunking
- Split cleaned text into overlapping word-based chunks (500 words, 50-word overlap).
- Assign each chunk a unique ID (`{entity_name}_chunk_{i}`).
- Preserve entity metadata (name, type) with every chunk.

### F3 — Local Embedding
- Generate dense vector embeddings for every chunk using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Embeddings computed entirely on-device; no external API call required.

### F4 — Vector Storage & Retrieval
- Persist embeddings and metadata in ChromaDB (`PersistentClient`).
- Support metadata-filtered search (restrict to "person" or "place" when query type is known).
- Return top-5 chunks by cosine similarity.

### F5 — Query Classification
- Keyword-matching pre-classifier determines whether a query is about a person, a place, or both.
- Applied before embedding to enable metadata-filtered retrieval and reduce noise.

### F6 — Answer Generation
- Build a structured prompt containing retrieved context and user question.
- Call Ollama HTTP API (`llama3.2:3b`) for non-streaming generation.
- Prompt instructs the model to answer only from context and say "I don't know" otherwise.
- Handle Ollama connection failures gracefully with a user-friendly error message.

### F7 — Streamlit Chat UI
- Persistent chat history across messages within a session.
- Sidebar: ingestion statistics, model info, Ollama status, show/hide sources toggle, clear chat button.
- Expandable "Retrieved sources" section showing entity name, type, distance, and a text snippet.
- "Thinking…" spinner during generation.

### F8 — SQLite Metadata Store
- Track each ingested entity: name, type, Wikipedia URL, chunk count, ingestion timestamp.
- Drive sidebar statistics (total people, places, chunks).
- Enable `document_exists()` check for incremental ingestion.

---

## 4. Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| Language | Python 3.10+ |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | ChromaDB ≥ 0.4.0 (PersistentClient, cosine space) |
| LLM runtime | Ollama (`llama3.2:3b`) via HTTP API |
| UI framework | Streamlit ≥ 1.28.0 |
| HTTP client | `requests` ≥ 2.31.0 (Ollama API); `urllib` (Wikipedia API) |
| Relational DB | SQLite (stdlib `sqlite3`) |
| Chunk size | 500 words with 50-word overlap |
| Top-K results | 5 chunks per query |
| External dependencies at inference | None (fully offline after ingestion) |
| Wikipedia source | MediaWiki REST API (`action=query&prop=extracts&explaintext=1`) |

---

## 5. Success Metrics

| Metric | Target |
|--------|--------|
| Ingestion success rate | ≥ 95% of 40 entities ingested without error |
| Answer relevance (manual spot-check) | Answers correctly reference context for ≥ 80% of test queries |
| Hallucination rate | 0% fabricated facts (system says "I don't know" if not in context) |
| Time-to-first-token | < 30 s on Apple M-series or modern x86 laptop |
| Ingestion runtime | < 10 minutes for all 40 entities on a standard laptop |
| UI responsiveness | Page renders in < 2 s; spinner shown for long operations |

---

## 6. Non-Functional Requirements

### NFR1 — Fully Local
All compute (embedding, retrieval, generation) runs on the user's machine. No data is sent to any external API at inference time. Internet access is only required during the initial `python ingest.py` run to fetch Wikipedia content.

### NFR2 — Reproducibility
Any user with Python 3.10+, `pip install -r requirements.txt`, and Ollama installed can reproduce the full system by following the README.

### NFR3 — Privacy
No user queries, responses, or retrieved chunks are logged to any external service.

### NFR4 — Modularity
Each component (ingest, retrieve, generate, database) is a separate module with a clean public API, enabling independent testing and replacement.

### NFR5 — Graceful Degradation
If Ollama is not running, the UI displays a clear error message rather than crashing. If a Wikipedia article is unavailable, ingestion skips it and logs the error without aborting the entire run.

### NFR6 — No Hallucination Guardrail
The prompt explicitly instructs the LLM to answer only from provided context and to say "I don't know based on my available information" when the context is insufficient.

---

## 7. Out of Scope (v1.0)

- Multi-document citations with inline footnotes.
- User authentication or multi-user sessions.
- Custom entity addition via the UI.
- Re-ranking (cross-encoder) of retrieved chunks.
- Streaming token-by-token output in the UI.
- Conversation memory beyond the current Streamlit session.
- Support for languages other than English.
