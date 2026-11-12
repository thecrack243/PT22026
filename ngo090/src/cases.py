"""Investigation cases: the durable record of what an analyst did with an alert.

The dashboard itself is stateless: the live feed is trimmed, replay positions are
session state and a browser refresh forgets everything. A case is the opposite --
it is written to SQLite the moment the analyst opens it and it is still there
after a restart, which is what makes an investigation auditable.

One row of the `cases` table is one case. The store is deliberately small and
plain: a connection per operation (Streamlit reruns in worker threads, and a
shared connection would need locking), WAL mode so a read never blocks a write,
and no ORM.

Statuses and priorities are fixed vocabularies; the store rejects anything else
so a typo cannot create a state the UI does not know how to show.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("open", "investigating", "confirmed_fraud", "false_positive", "closed")
PRIORITIES = ("low", "medium", "high", "urgent")

STATUS_LABELS = {
    "open": "Open",
    "investigating": "Investigating",
    "confirmed_fraud": "Confirmed fraud",
    "false_positive": "False positive",
    "closed": "Closed",
}
PRIORITY_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "urgent": "Urgent"}
DECISION_LABELS = {"approve": "Approved", "review": "Sent to review", "block": "Blocked"}

ACTIVE = ("open", "investigating")          # a case still being worked on
CLOSED = ("confirmed_fraud", "false_positive", "closed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL,
    risk        REAL    NOT NULL,
    decision    TEXT    NOT NULL,
    explanation TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'open',
    notes       TEXT    NOT NULL DEFAULT '',
    priority    TEXT    NOT NULL DEFAULT 'medium',
    source      TEXT    NOT NULL DEFAULT 'live',
    amount      REAL    NOT NULL DEFAULT 0,
    tx_type     TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_txn    ON cases(txn_id);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
"""
_COLUMNS = ("id", "txn_id", "risk", "decision", "explanation", "status", "notes",
            "priority", "source", "amount", "tx_type", "created_at", "updated_at")


def case_id(number: int) -> str:
    """Human-readable id: row 7 is 'C-0007'."""
    return f"C-{int(number):04d}"


def parse_case_id(value: str | int) -> int:
    """Accept 'C-0007', 'c7' or 7 and return the row number."""
    text = str(value).strip().upper().lstrip("C").lstrip("-")
    if not text.isdigit():
        raise ValueError(f"Not a case id: {value!r}")
    return int(text)


def priority_for(risk: float, decision: str = "") -> str:
    """Starting priority, from the score the engine gave the transaction."""
    if decision == "block" or risk >= 96:
        return "urgent"
    if risk >= 92:
        return "high"
    if risk >= 85:
        return "medium"
    return "low"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}. Use one of {', '.join(STATUSES)}.")
    return status


def _check_priority(priority: str) -> str:
    if priority not in PRIORITIES:
        raise ValueError(f"Unknown priority {priority!r}. Use one of {', '.join(PRIORITIES)}.")
    return priority


@dataclass(frozen=True)
class Case:
    """One investigation. Frozen: an edit goes through the store and returns a new copy."""

    number: int                        # row id, shown as case_id(number)
    txn_id: int
    risk: float
    decision: str
    explanation: str = ""
    status: str = "open"
    notes: str = ""
    priority: str = "medium"
    source: str = "live"               # live / replay / testing
    amount: float = 0.0
    tx_type: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def case_id(self) -> str:
        return case_id(self.number)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def priority_label(self) -> str:
        return PRIORITY_LABELS[self.priority]

    @property
    def decision_label(self) -> str:
        return DECISION_LABELS.get(self.decision, self.decision.title())


class CaseStore:
    """SQLite-backed store of investigation cases.

    The path is created on demand (including its folder), so a fresh clone works
    without any setup step. `:memory:` is accepted for tests.
    """

    def __init__(self, path: str | Path = "data/cases.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory = None if self.path != ":memory:" else sqlite3.connect(":memory:")
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        if self._memory is not None:
            return self._memory
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        if self._memory is None:
            conn.close()

    def _row_to_case(self, row: sqlite3.Row) -> Case:
        return Case(number=row["id"], txn_id=row["txn_id"], risk=row["risk"],
                    decision=row["decision"], explanation=row["explanation"],
                    status=row["status"], notes=row["notes"], priority=row["priority"],
                    source=row["source"], amount=row["amount"], tx_type=row["tx_type"],
                    created_at=row["created_at"], updated_at=row["updated_at"])

    def _select(self, where: str = "", params: tuple = (), order: str = "updated_at DESC",
                limit: int | None = None) -> list[Case]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM cases"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order}"
        if limit:
            sql += " LIMIT ?"
            params = params + (int(limit),)
        conn = self._connect()
        try:
            return [self._row_to_case(r) for r in conn.execute(sql, params)]
        finally:
            if self._memory is None:
                conn.close()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def get(self, value: str | int) -> Case | None:
        """One case by id ('C-0007' or 7). None when it does not exist."""
        found = self._select("id = ?", (parse_case_id(value),), limit=1)
        return found[0] if found else None

    def open_for(self, txn_id: int) -> Case | None:
        """The case still being worked on for a transaction, if any.

        Used before creating one, so clicking twice does not open two cases for
        the same alert. Closed cases do not block a new investigation.
        """
        active = ",".join("?" * len(ACTIVE))
        found = self._select(f"txn_id = ? AND status IN ({active})", (int(txn_id), *ACTIVE),
                             order="updated_at DESC", limit=1)
        return found[0] if found else None

    def for_transaction(self, txn_id: int) -> list[Case]:
        return self._select("txn_id = ?", (int(txn_id),))

    def list(self, status: str | None = None, priority: str | None = None,
             search: str = "", limit: int = 500) -> list[Case]:
        """Filtered list, most recently updated first. `search` matches case or transaction id."""
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(_check_status(status))
        if priority:
            clauses.append("priority = ?")
            params.append(_check_priority(priority))
        raw = str(search).strip()
        if raw:
            upper = raw.upper()
            if upper.startswith("C"):                   # 'C-0007' only looks at case ids
                digits = upper.lstrip("C").lstrip("-").strip()
                if not digits.isdigit():
                    return []
                clauses.append("id = ?")
                params.append(int(digits))
            elif raw.startswith("#"):                   # '#1234' only looks at transaction ids
                digits = raw.lstrip("#").strip()
                if not digits.isdigit():
                    return []
                clauses.append("txn_id = ?")
                params.append(int(digits))
            else:                                       # a bare number matches either id
                if not raw.isdigit():
                    return []
                clauses.append("(id = ? OR txn_id = ?)")
                params.extend([int(raw), int(raw)])
        where = " AND ".join(clauses)
        return self._select(where, tuple(params), order="updated_at DESC", limit=limit)

    def counts(self) -> dict[str, int]:
        """Cases per status (every status present, zeros included) plus 'all' and 'active'."""
        out = {s: 0 for s in STATUSES}
        conn = self._connect()
        try:
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM cases GROUP BY status"):
                out[row["status"]] = row["n"]
        finally:
            if self._memory is None:
                conn.close()
        out["all"] = sum(out[s] for s in STATUSES)
        out["active"] = sum(out[s] for s in ACTIVE)
        return out

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def create(self, txn_id: int, risk: float, decision: str, explanation: str = "",
               *, priority: str | None = None, source: str = "live", amount: float = 0.0,
               tx_type: str = "", status: str = "open", notes: str = "") -> Case:
        """Insert a new case and return it (with its generated id and timestamps)."""
        risk = float(risk)
        priority = _check_priority(priority or priority_for(risk, decision))
        status = _check_status(status)
        stamp = _now()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO cases (txn_id, risk, decision, explanation, status, notes, priority,"
                " source, amount, tx_type, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (int(txn_id), risk, str(decision), str(explanation), status, str(notes), priority,
                 str(source), float(amount), str(tx_type), stamp, stamp),
            )
            conn.commit()
            number = int(cur.lastrowid)
        finally:
            if self._memory is None:
                conn.close()
        return Case(number=number, txn_id=int(txn_id), risk=risk, decision=str(decision),
                    explanation=str(explanation), status=status, notes=str(notes),
                    priority=priority, source=str(source), amount=float(amount),
                    tx_type=str(tx_type), created_at=stamp, updated_at=stamp)

    def update(self, value: str | int, *, status: str | None = None, notes: str | None = None,
               priority: str | None = None) -> Case:
        """Change status, notes and/or priority. Only real changes touch updated_at."""
        case = self.get(value)
        if case is None:
            raise KeyError(f"Case {value!r} does not exist.")
        new_status = _check_status(status) if status is not None else case.status
        new_priority = _check_priority(priority) if priority is not None else case.priority
        new_notes = case.notes if notes is None else str(notes)

        changed = (new_status, new_priority, new_notes) != (case.status, case.priority, case.notes)
        updated = _now() if changed else case.updated_at
        conn = self._connect()
        try:
            conn.execute("UPDATE cases SET status = ?, notes = ?, priority = ?, updated_at = ? WHERE id = ?",
                         (new_status, new_notes, new_priority, updated, case.number))
            conn.commit()
        finally:
            if self._memory is None:
                conn.close()
        return replace(case, status=new_status, notes=new_notes, priority=new_priority,
                       updated_at=updated)

    def set_status(self, value: str | int, status: str) -> Case:
        """Shortcut for the buttons that move a case through its life cycle."""
        return self.update(value, status=status)

    def delete(self, value: str | int) -> bool:
        """Remove a case (kept for tests and for cleaning up mistakes)."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM cases WHERE id = ?", (parse_case_id(value),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            if self._memory is None:
                conn.close()

    def close(self) -> None:
        """Release the in-memory connection (file-backed stores open per call)."""
        if self._memory is not None:
            self._memory.close()
            self._memory = None
