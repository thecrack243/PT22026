"""The conversation between an analyst and the assistant, one case at a time.

This is a record, not a chat log in the disposable sense: what an analyst asked
and what the assistant answered is part of the case file. Two properties follow
from that and shape the module:

    * append-only -- there is no update and no delete. A wrong answer stays
      visible next to the question that produced it, which is what makes the
      transcript worth reading during an audit.
    * per case -- every row carries the case it belongs to, so the transcript
      travels with the investigation and cannot leak into another one.

A message may carry a structured `action`: what the assistant *did* (appended a
note) or what it *asked for* (filed a proposal, by number). The text alone would
be ambiguous later; the action is the machine-readable half of the same turn.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.cases import parse_case_id

ROLES = ("analyst", "assistant", "system")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    text       TEXT    NOT NULL DEFAULT '',
    action     TEXT    NOT NULL DEFAULT '{}',
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_case ON messages(case_id, id);
"""

_COLUMNS = ("id", "case_id", "role", "text", "action", "created_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Message:
    """One turn: who said it, what they said, and what it did."""
    number: int
    case_number: int
    role: str
    text: str
    action: dict | None = None
    created_at: str = ""

    @property
    def case_id(self) -> str:
        from src.cases import case_id as render

        return render(self.case_number)

    @property
    def from_analyst(self) -> bool:
        return self.role == "analyst"

    def to_dict(self) -> dict:
        return {"message": int(self.number), "case": self.case_id, "role": self.role,
                "text": self.text, "action": self.action or {}, "at": self.created_at}


class ChatStore:
    """SQLite-backed transcript. Same conventions as the other stores.

    `:memory:` is supported and shared for the life of the instance, which is
    what the tests use.
    """

    def __init__(self, path: str | Path = "data/chat.db"):
        self.path = str(path)
        self._memory = None if self.path != ":memory:" else sqlite3.connect(":memory:")
        if self._memory is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            if self._memory is None:
                conn.close()

    # -- plumbing ------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None:
            conn = self._memory
        else:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    @staticmethod
    def _row(row: sqlite3.Row) -> Message:
        try:
            action = json.loads(row["action"] or "{}")
        except json.JSONDecodeError:              # a hand-edited database should not crash the page
            action = {}
        return Message(number=row["id"], case_number=row["case_id"], role=row["role"],
                       text=row["text"], action=action or None, created_at=row["created_at"])

    # -- reading -------------------------------------------------------- #
    def history(self, case: int | str, limit: int = 200) -> list[Message]:
        """The transcript, oldest first: a conversation reads downwards."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM messages WHERE case_id = ?"
                " ORDER BY id DESC LIMIT ?", (parse_case_id(case), int(limit))).fetchall()
        finally:
            if self._memory is None:
                conn.close()
        return [self._row(row) for row in reversed(rows)]

    def last(self, case: int | str) -> Message | None:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM messages WHERE case_id = ?"
                " ORDER BY id DESC LIMIT 1", (parse_case_id(case),)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return None if row is None else self._row(row)

    def count(self, case: int | str) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE case_id = ?",
                               (parse_case_id(case),)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        return int(row[0])

    # -- writing -------------------------------------------------------- #
    def append(self, case: int | str, role: str, text: str,
               action: dict | None = None) -> Message:
        """Add one turn. Nothing here can rewrite an earlier one."""
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}. Use one of {', '.join(ROLES)}.")
        number = parse_case_id(case)
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO messages (case_id, role, text, action, created_at)"
                " VALUES (?,?,?,?,?)",
                (number, role, str(text), json.dumps(action or {}, sort_keys=True), _now()))
            conn.commit()
            fresh = int(cur.lastrowid)
        finally:
            if self._memory is None:
                conn.close()
        conn = self._connect()
        try:
            row = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM messages WHERE id = ?",
                               (fresh,)).fetchone()
        finally:
            if self._memory is None:
                conn.close()
        if row is None:                            # cannot happen; keeps the return type honest
            raise RuntimeError("The message could not be read back after writing it.")
        return self._row(row)
