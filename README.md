# Wikipedia RAG Assistant

A fully local, ChatGPT-style question-answering system that retrieves factual information about famous people and places from Wikipedia and generates grounded answers using a locally-running language model.

---

## System Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│              Streamlit UI (app.py)      │
└─────────────────┬───────────────────────┘
                  │
         ┌────────▼────────┐
         │  Query Classifier│  (retriever.py)
         │  person/place/  │
         │  both           │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Sentence-Trans │  embed query
         │  (MiniLM-L6-v2) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   ChromaDB      │  cosine similarity search
         │  (local disk)   │  optional metadata filter
         └────────┬────────┘
                  │ top-5 chunks
         ┌────────▼────────┐
         │  Prompt Builder │  (generator.py)
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Ollama HTTP API│  llama3.2:3b
         │  /api/generate  │
         └────────┬────────┘
                  │
            Answer (text)
```

**Storage layers**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Vector store | ChromaDB (PersistentClient) | Chunk embeddings + metadata |
| Relational | SQLite | Ingestion metadata (entity name, type, URL, chunk count) |

---

## Installation

### 1. Clone / navigate to the project directory

```bash
cd /path/to/aihw3
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> The first run will also download the `all-MiniLM-L6-v2` model (~90 MB) automatically via sentence-transformers.

---

## Ollama Setup

Ollama runs the LLM entirely on your machine.

### macOS

```bash
brew install ollama
ollama pull llama3.2:3b
ollama serve          # keep this running in a separate terminal
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve
```

### Windows

Download the installer from <https://ollama.com/download> and follow the prompts.

```powershell
ollama pull llama3.2:3b
ollama serve
```

> `ollama serve` must be running before you start the Streamlit app.

---

## Ingesting Data

Fetch all 40 Wikipedia articles (20 people + 20 places), chunk them, embed them, and store them locally:

```bash
python ingest.py
```

To wipe existing data and re-ingest everything from scratch:

```bash
python ingest.py --reset
```

Expected output (abbreviated):

```
============================================================
  Wikipedia RAG Assistant — Ingestion Pipeline
============================================================

Loading embedding model 'all-MiniLM-L6-v2' ...
Model loaded.

── Ingesting 20 people ──────────────────────────────
  [FETCH] Albert Einstein ... 42 chunks ✓
  [FETCH] Marie Curie ... 38 chunks ✓
  ...

── Ingesting 20 places ──────────────────────────────
  [FETCH] Eiffel Tower ... 25 chunks ✓
  ...

============================================================
  Ingestion complete.
  People ingested : 20/20
  Places ingested : 20/20
  Total chunks in ChromaDB: 1247
============================================================
```

---

## Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Example Queries

### People

- "Where was Albert Einstein born and what did he study?"
- "What awards did Marie Curie receive?"
- "When did Leonardo da Vinci live and what is he famous for?"
- "Describe Ada Lovelace's contribution to computing."
- "What was Nikola Tesla known for?"
- "How many World Cup goals has Cristiano Ronaldo scored?"
- "What genre of music is Taylor Swift known for?"
- "What was Mahatma Gandhi's role in Indian independence?"
- "Tell me about Stephen Hawking's scientific contributions."
- "How did Frida Kahlo's life influence her paintings?"

### Places

- "How tall is the Eiffel Tower and when was it built?"
- "Where is the Great Wall of China located?"
- "Who built the Taj Mahal and why?"
- "What country is Machu Picchu in?"
- "Describe the history of the Colosseum."
- "What religion is the Hagia Sophia associated with?"
- "When was the Statue of Liberty dedicated?"
- "Where are the Pyramids of Giza located?"
- "What is the height of Mount Everest?"
- "What is Stonehenge and what is its purpose?"

### Cross-entity / broad

- "Compare Einstein and Newton's contributions to physics."
- "Which famous landmarks are located in Europe?"

---

## Project Structure

```
aihw3/
├── app.py              # Streamlit chat UI
├── ingest.py           # Data fetching, chunking, embedding, storage
├── retriever.py        # Query classification + ChromaDB retrieval
├── generator.py        # Ollama LLM answer generation
├── database.py         # SQLite helpers
├── config.py           # Centralised constants
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── product_prd.md      # Product Requirements Document
├── recommendation.md   # Production deployment recommendations
├── chroma_db/          # ChromaDB persistent storage (auto-created)
└── rag_metadata.db     # SQLite database (auto-created)
```

---

## Design Decisions

### Why urllib instead of the `wikipedia` library?

The homework specification requires using the MediaWiki REST API directly with `urllib`. This avoids the third-party `wikipedia` package and gives full control over what fields are fetched (plain text extracts, redirect following, etc.).

### Why sentence-transformers (local) instead of an API?

All embeddings are generated locally using `all-MiniLM-L6-v2`. This ensures zero latency dependency on an external API, no cost per embedding, and complete data privacy.

### Why ChromaDB?

ChromaDB is a lightweight, file-based vector database that runs entirely in-process with no server required. Its `PersistentClient` stores data on disk across runs, and it supports metadata filtering — allowing us to restrict retrieval to people or places when the query type is known.

### Why SQLite alongside ChromaDB?

ChromaDB is optimised for vector search, not relational queries. SQLite stores structured metadata (entity name, type, URL, chunk count, ingestion timestamp) that powers the sidebar stats panel and the `document_exists()` check during incremental ingestion.

### Why Ollama (llama3.2:3b)?

The system is designed to run 100% locally — no OpenAI API key or internet connection required at inference time. `llama3.2:3b` is a capable 3-billion-parameter model that runs comfortably on a modern laptop (including Apple Silicon via Metal).

### Chunking strategy

Word-based chunking with 500-word windows and 50-word overlap ensures that no single chunk is too short (losing context) or too long (diluting relevance). The overlap prevents information at chunk boundaries from being missed.

### Query classification

Before embedding the query, a fast keyword-matching step classifies it as about a person, a place, or both. When the type is specific, ChromaDB's `where` filter restricts the search to that entity type, reducing noise in the retrieved context.
