"""
generator.py - Answer generation via the Ollama HTTP API.

Uses requests.post to call http://localhost:11434/api/generate (non-streaming).
"""

from __future__ import annotations

import json

import requests

from config import LLM_MODEL, OLLAMA_BASE_URL

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about famous people and places. "
    "Use ONLY the following context to answer the question. "
    "If the answer is not in the context, say "
    "\"I don't know based on my available information.\" "
    "Do not make up information."
)


def _build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the full prompt string from retrieved context chunks and the user query.

    Parameters
    ----------
    query          : user's question
    context_chunks : list of dicts returned by retriever.retrieve()
                     Each dict must have at least "entity_name" and "text" keys.
    """
    context_parts: list[str] = []
    for chunk in context_chunks:
        entity = chunk.get("entity_name", "Unknown")
        text = chunk.get("text", "").strip()
        context_parts.append(f"[Entity: {entity}]\n{text}")

    context_str = "\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


# ── Ollama call ───────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str = LLM_MODEL,
) -> str:
    """
    Generate an answer for *query* using *context_chunks* as grounding context.

    Parameters
    ----------
    query          : user's natural-language question
    context_chunks : retrieved chunks from ChromaDB (list of dicts)
    model          : Ollama model tag (default from config)

    Returns
    -------
    Answer string.  Returns a graceful error message if Ollama is unavailable.
    """
    if not context_chunks:
        return (
            "I don't know based on my available information. "
            "No relevant context was retrieved for your question."
        )

    prompt = _build_prompt(query, context_chunks)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temperature → more factual
            "num_predict": 512,   # max tokens in response
        },
    }

    api_url = f"{OLLAMA_BASE_URL}/api/generate"

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=120,  # generation can take a while
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. "
            "Please start it with `ollama serve` and try again."
        )
    except requests.exceptions.Timeout:
        return (
            "The request to Ollama timed out. "
            "The model may still be loading — please try again in a moment."
        )
    except requests.exceptions.HTTPError as exc:
        return f"Ollama returned an error: {exc}"
    except Exception as exc:
        return f"Unexpected error communicating with Ollama: {exc}"

    try:
        data = response.json()
        answer = data.get("response", "").strip()
        if not answer:
            return "Ollama returned an empty response. Please try again."
        return answer
    except json.JSONDecodeError:
        return "Could not parse Ollama's response. Please try again."


# ── Convenience: check if Ollama is reachable ─────────────────────────────────

def ollama_is_available(model: str = LLM_MODEL) -> tuple[bool, str]:
    """
    Ping the Ollama API to check availability.

    Returns
    -------
    (True, "")            if reachable and model is available
    (False, error_msg)    otherwise
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(m.startswith(model.split(":")[0]) for m in models):
            return False, (
                f"Model '{model}' not found in Ollama. "
                f"Run: ollama pull {model}"
            )
        return True, ""
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except Exception as exc:
        return False, f"Could not reach Ollama: {exc}"
