"""
database.py - SQLite helper for storing document/ingestion metadata.
"""

from __future__ import annotations

import sqlite3
import os
from datetime import datetime
from config import SQLITE_DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the documents table if it doesn't already exist."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT    NOT NULL UNIQUE,
                entity_type TEXT    NOT NULL CHECK(entity_type IN ('person', 'place')),
                url         TEXT    NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                ingested_at TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_document(
    entity_name: str,
    entity_type: str,
    url: str,
    chunk_count: int,
    ingested_at: str | None = None,
) -> None:
    """
    Insert or replace a document record.

    Parameters
    ----------
    entity_name : str
    entity_type : str  "person" or "place"
    url         : str  Wikipedia URL
    chunk_count : int  number of text chunks stored in ChromaDB
    ingested_at : str  ISO timestamp; defaults to now
    """
    if ingested_at is None:
        ingested_at = datetime.utcnow().isoformat()

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO documents (entity_name, entity_type, url, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entity_name) DO UPDATE SET
                entity_type = excluded.entity_type,
                url         = excluded.url,
                chunk_count = excluded.chunk_count,
                ingested_at = excluded.ingested_at
            """,
            (entity_name, entity_type, url, chunk_count, ingested_at),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_documents() -> list[dict]:
    """Return all document records as a list of dicts."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY entity_type, entity_name"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def document_exists(entity_name: str) -> bool:
    """Return True if the entity has already been ingested."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE entity_name = ?", (entity_name,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_stats() -> dict:
    """
    Return aggregate statistics.

    Returns
    -------
    dict with keys: total_people, total_places, total_chunks, total_documents
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                              AS total_documents,
                COALESCE(SUM(entity_type = 'person'), 0)              AS total_people,
                COALESCE(SUM(entity_type = 'place'),  0)              AS total_places,
                COALESCE(SUM(chunk_count), 0)                         AS total_chunks
            FROM documents
            """
        ).fetchone()
        return dict(row) if row else {
            "total_documents": 0,
            "total_people": 0,
            "total_places": 0,
            "total_chunks": 0,
        }
    finally:
        conn.close()


def delete_all() -> None:
    """Delete all rows from the documents table (used on --reset)."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM documents")
        conn.commit()
    finally:
        conn.close()
