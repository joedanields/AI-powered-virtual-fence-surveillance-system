"""
Database module — SQLite-backed intrusion event logger.

Provides thread-safe CRUD operations for intrusion events.
Designed for concurrent access from the detection pipeline and the web dashboard.
"""

import sqlite3
import threading
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class IntrusionDatabase:
    """Thread-safe SQLite database for intrusion event persistence."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS intrusion_events (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            camera_id   TEXT    NOT NULL,
            zone_name   TEXT    DEFAULT 'default',
            screenshot_path TEXT,
            video_path  TEXT,
            num_persons INTEGER DEFAULT 1,
            status      TEXT    DEFAULT 'new',
            notes       TEXT
        );
    """

    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_timestamp ON intrusion_events(timestamp);
    """

    def __init__(self, config: DatabaseConfig):
        self._db_path = config.db_path
        self._lock = threading.Lock()
        self._init_database()
        logger.info("IntrusionDatabase initialised → %s", self._db_path)

    # ── connection helper ──────────────────────
    @contextmanager
    def _get_connection(self):
        """Yields a connection with WAL journal mode for concurrent reads."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create tables and indices if they don't exist."""
        with self._lock, self._get_connection() as conn:
            conn.execute(self._CREATE_TABLE)
            conn.execute(self._CREATE_INDEX)

    # ── INSERT ─────────────────────────────────
    def log_event(
        self,
        camera_id: str,
        screenshot_path: str = "",
        video_path: str = "",
        zone_name: str = "default",
        num_persons: int = 1,
        status: str = "new",
        notes: str = "",
    ) -> int:
        """
        Insert a new intrusion event.

        Returns:
            event_id of the newly created row.
        """
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO intrusion_events
                    (timestamp, camera_id, zone_name, screenshot_path,
                     video_path, num_persons, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, camera_id, zone_name, screenshot_path, video_path,
                 num_persons, status, notes),
            )
            event_id = cursor.lastrowid
        logger.info("Event #%d logged at %s [camera=%s]", event_id, ts, camera_id)
        return event_id

    # ── SELECT ─────────────────────────────────
    def get_events(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve intrusion events with optional filters, newest first."""
        query = "SELECT * FROM intrusion_events WHERE 1=1"
        params: list = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single event by its primary key."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM intrusion_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def count_events(self, camera_id: Optional[str] = None) -> int:
        """Total event count (optionally filtered by camera)."""
        query = "SELECT COUNT(*) FROM intrusion_events"
        params: list = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
        with self._get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    # ── UPDATE ─────────────────────────────────
    def update_status(self, event_id: int, status: str) -> bool:
        """Mark an event as acknowledged / resolved / etc."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE intrusion_events SET status = ? WHERE event_id = ?",
                (status, event_id),
            )
        return cursor.rowcount > 0

    # ── DELETE ─────────────────────────────────
    def delete_event(self, event_id: int) -> bool:
        """Remove an event by ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM intrusion_events WHERE event_id = ?",
                (event_id,),
            )
        return cursor.rowcount > 0

    # ── STATISTICS ─────────────────────────────
    def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Intrusion counts grouped by date for the last N days."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM intrusion_events
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                """,
                (f"-{days} days",),
            ).fetchall()
        return [dict(r) for r in rows]
