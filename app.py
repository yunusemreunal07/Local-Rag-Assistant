"""
app.py - Streamlit chat interface for the Wikipedia RAG Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st

from config import EMBEDDING_MODEL, LLM_MODEL, TOP_K
from database import get_stats, init_db
from generator import generate_answer, ollama_is_available
from retriever import classify_query, retrieve

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Session-state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, sources?}
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


# ── Helper: initialise DB ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_db():
    init_db()


_init_db()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings & Status")

    # Ingestion stats
    st.subheader("📊 Ingestion Status")
    try:
        stats = get_stats()
        col1, col2 = st.columns(2)
        col1.metric("People", stats.get("total_people", 0))
        col2.metric("Places", stats.get("total_places", 0))
        st.metric("Total Chunks", stats.get("total_chunks", 0))
        if stats.get("total_documents", 0) == 0:
            st.warning("No data ingested yet.\nRun: `python ingest.py`")
    except Exception as exc:
        st.error(f"DB error: {exc}")

    st.divider()

    # Model info
    st.subheader("🤖 Model Info")
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Top-K:** `{TOP_K}`")

    # Ollama status
    available, err_msg = ollama_is_available()
    if available:
        st.success("Ollama: connected ✓")
    else:
        st.error(f"Ollama: {err_msg}")

    st.divider()

    # Show/hide sources toggle
    st.subheader("🔍 Display Options")
    st.session_state.show_sources = st.toggle(
        "Show source chunks", value=st.session_state.show_sources
    )

    # Clear chat
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Wikipedia RAG Assistant\n"
        "Powered by sentence-transformers + ChromaDB + Ollama"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📚 Wikipedia RAG Assistant")
st.caption(
    "Ask me anything about the famous people and places in my knowledge base."
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources if available and toggle is on
        if (
            msg["role"] == "assistant"
            and st.session_state.show_sources
            and msg.get("sources")
        ):
            with st.expander("📄 Retrieved sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**{i}. {src['entity_name']}** "
                        f"_(type: {src['type']}, distance: {src['distance']})_"
                    )
                    snippet = src["text"][:300].replace("\n", " ")
                    st.caption(f"…{snippet}…")
                    st.divider()


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about a person or place…")

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Check data availability
    try:
        stats = get_stats()
        if stats.get("total_documents", 0) == 0:
            answer = (
                "The knowledge base is empty. "
                "Please run `python ingest.py` first to ingest Wikipedia data."
            )
            sources: list[dict] = []
            with st.chat_message("assistant"):
                st.warning(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.stop()
    except Exception:
        pass

    # RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # 1. Classify query
            query_type = classify_query(user_input)

            # 2. Retrieve relevant chunks
            try:
                chunks = retrieve(user_input, query_type=query_type, top_k=TOP_K)
            except Exception as exc:
                chunks = []
                st.error(f"Retrieval error: {exc}")

            # 3. Generate answer
            if not ollama_is_available()[0]:
                answer = (
                    "Ollama is not running. "
                    "Please start it with `ollama serve` and ensure "
                    f"`{LLM_MODEL}` is pulled."
                )
            else:
                answer = generate_answer(user_input, chunks)

        # Display answer
        st.markdown(answer)

        # Show sources (expandable)
        if st.session_state.show_sources and chunks:
            with st.expander("📄 Retrieved sources", expanded=False):
                for i, src in enumerate(chunks, 1):
                    st.markdown(
                        f"**{i}. {src['entity_name']}** "
                        f"_(type: {src['type']}, distance: {src['distance']})_"
                    )
                    snippet = src["text"][:300].replace("\n", " ")
                    st.caption(f"…{snippet}…")
                    st.divider()

    # Persist to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": chunks,
        }
    )
