"""
SQLite-backed journal store — shared by app.py and api.py.
Database lives at outputs/journal.db (persisted via Docker volume).
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "outputs" / "journal.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                transcription TEXT NOT NULL,
                emotions  TEXT NOT NULL,
                dominant  TEXT NOT NULL
            )
        """)


def add_entry(entry: dict) -> dict:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO entries (timestamp, transcription, emotions, dominant) VALUES (?, ?, ?, ?)",
            (entry["timestamp"], entry["transcription"],
             json.dumps(entry["emotions"]), entry["dominant"]),
        )
    return entry


def get_entries(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "timestamp": r["timestamp"],
            "transcription": r["transcription"],
            "emotions": json.loads(r["emotions"]),
            "dominant": r["dominant"],
        }
        for r in reversed(rows)
    ]


def clear_entries() -> int:
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.execute("DELETE FROM entries")
    return count


def total_entries() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


_init()
