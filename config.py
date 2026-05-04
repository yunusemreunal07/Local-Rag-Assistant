"""
config.py - Central configuration constants for the Wikipedia RAG Assistant.
"""

import os

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # words of overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5               # number of chunks to retrieve per query

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model
LLM_MODEL = "llama3.2:3b"              # Ollama model for answer generation
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Storage paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "rag_metadata.db")
CHROMA_COLLECTION_NAME = "wikipedia_rag"

# ── Wikipedia API ─────────────────────────────────────────────────────────────
WIKIPEDIA_API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query"
    "&prop=extracts"
    "&explaintext=1"
    "&titles={title}"
    "&format=json"
    "&redirects=1"
)
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/wiki/{title}"

# ── Entities to ingest ────────────────────────────────────────────────────────
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Galileo Galilei",
    "Cleopatra",
    "Napoleon Bonaparte",
    "Abraham Lincoln",
    "Winston Churchill",
    "Nelson Mandela",
    "Stephen Hawking",
    "Mahatma Gandhi",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Stonehenge",
    "Angkor Wat",
    "Petra, Jordan",
    "Chichen Itza",
    "Sydney Opera House",
    "Big Ben",
    "Acropolis of Athens",
    "Niagara Falls",
    "Amazon River",
    "Mount Fuji",
]

# ── Query classifier word lists (lowercase) ───────────────────────────────────
KNOWN_PEOPLE_KEYWORDS = [p.lower() for p in PEOPLE] + [
    "einstein", "curie", "da vinci", "shakespeare", "lovelace",
    "tesla", "messi", "ronaldo", "swift", "kahlo",
    "newton", "darwin", "galileo", "cleopatra", "napoleon",
    "lincoln", "churchill", "mandela", "hawking", "gandhi",
]

KNOWN_PLACES_KEYWORDS = [p.lower() for p in PLACES] + [
    "eiffel", "great wall", "taj mahal", "grand canyon", "machu picchu",
    "colosseum", "hagia sophia", "statue of liberty", "pyramids", "giza",
    "mount everest", "everest", "stonehenge", "angkor", "petra",
    "chichen itza", "sydney opera", "big ben", "acropolis", "niagara",
    "amazon river", "mount fuji", "fuji",
]
