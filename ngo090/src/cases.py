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

STATUSES = ("investigating", "confirmed_fraud", "false_positive")
PRIORITIES = ("low", "medium", "high", "urgent")

STATUS_LABELS = {
    "investigating": "Investigating",
    "confirmed_fraud": "Confirmed fraud",
    "false_positive": "False positive",
}
PRIORITY_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "urgent": "Urgent"}
DECISION_LABELS = {"approve": "Approved", "review": "Sent to review", "block": "Blocked"}

# A case is open (being worked on) or closed (a decision was made and recorded).
# The resolution (status above) and the state are independent: an open case can
# already name its resolution, a closed one always carries one.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      INTEGER NOT NULL,
    risk        REAL    NOT NULL,
    decision    TEXT    NOT NULL,
    explanation TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'investigating',
    closed      INTEGER NOT NULL DEFAULT 0,
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
_COLUMNS = ("id", "txn_id", "risk", "decision", "explanation", "status", "closed", "notes",
            "priority", "source", "amount", "tx_type", "created_at", "updated_at")

# Statuses written by the first version of the store (before the closed flag existed).
_LEGACY_OPEN = "open"
_LEGACY_CLOSED = "closed"


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
    status: str = "investigating"      # the resolution: investigating / confirmed / false positive
    closed: bool = False               # the state: still open, or done and archived
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
    def is_open(self) -> bool:
        return not self.closed

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def state_label(self) -> str:
        return "Closed" if self.closed else "Open"

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
        self._migrate(conn)
        conn.commit()
        if self._memory is None:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Bring a database written by an older version up to the current model.

        The first version had a single status column with 'open' and 'closed'
        values. The current model separates the resolution (status) from the
        state (closed flag), so old rows are remapped on startup.
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(cases)")}
        if "closed" not in cols:
            conn.execute("ALTER TABLE cases ADD COLUMN closed INTEGER NOT NULL DEFAULT 0")
        conn.execute(f"UPDATE cases SET status = 'investigating' WHERE status = ?", (_LEGACY_OPEN,))
        conn.execute("UPDATE cases SET closed = 1, status = 'investigating' WHERE status = ?",
                     (_LEGACY_CLOSED,))

    def _row_to_case(self, row: sqlite3.Row) -> Case:
        return Case(number=row["id"], txn_id=row["txn_id"], risk=row["risk"],
                    decision=row["decision"], explanation=row["explanation"],
                    status=row["status"], closed=bool(row["closed"]), notes=row["notes"],
                    priority=row["priority"], source=row["source"], amount=row["amount"],
                    tx_type=row["tx_type"], created_at=row["created_at"], updated_at=row["updated_at"])

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
        """The not-closed case for a transaction, if any.

        Used before creating one, so clicking twice does not open two cases for
        the same alert. Closed cases do not block a new investigation.
        """
        found = self._select("txn_id = ? AND closed = 0", (int(txn_id),),
                             order="updated_at DESC", limit=1)
        return found[0] if found else None

    def for_transaction(self, txn_id: int) -> list[Case]:
        return self._select("txn_id = ?", (int(txn_id),))

    def list(self, status: str | None = None, state: str | None = None,
             priority: str | None = None, search: str = "", limit: int = 500) -> list[Case]:
        """Filtered list, most recently updated first.

        `status` is the resolution, `state` is 'open' or 'closed'; they combine.
        `search` matches case or transaction id.
        """
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(_check_status(status))
        if state:
            state = str(state).strip().lower()
            if state not in ("open", "closed"):
                raise ValueError("State must be 'open' or 'closed'.")
            clauses.append("closed = ?")
            params.append(1 if state == "closed" else 0)
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
        """Pill and badge numbers.

        Per-resolution counts cover open cases only (what the analyst can still
        work on); 'closed' counts every closed case; 'open' is the sidebar badge;
        'all' is everything.
        """
        out = {s: 0 for s in STATUSES}
        closed = 0
        conn = self._connect()
        try:
            for row in conn.execute("SELECT status, closed, COUNT(*) AS n FROM cases GROUP BY status, closed"):
                if row["closed"]:
                    closed += row["n"]
                elif row["status"] in out:
                    out[row["status"]] = row["n"]
        finally:
            if self._memory is None:
                conn.close()
        out["closed"] = closed
        out["open"] = sum(out[s] for s in STATUSES)
        out["all"] = out["open"] + closed
        return out

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def create(self, txn_id: int, risk: float, decision: str, explanation: str = "",
               *, priority: str | None = None, source: str = "live", amount: float = 0.0,
               tx_type: str = "", status: str = "investigating", notes: str = "",
               closed: bool = False) -> Case:
        """Insert a new case (default: open and investigating) and return it."""
        risk = float(risk)
        priority = _check_priority(priority or priority_for(risk, decision))
        status = _check_status(status)
        stamp = _now()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO cases (txn_id, risk, decision, explanation, status, closed, notes,"
                " priority, source, amount, tx_type, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (int(txn_id), risk, str(decision), str(explanation), status,
                 1 if closed else 0, str(notes), priority,
                 str(source), float(amount), str(tx_type), stamp, stamp),
            )
            conn.commit()
            number = int(cur.lastrowid)
        finally:
            if self._memory is None:
                conn.close()
        return Case(number=number, txn_id=int(txn_id), risk=risk, decision=str(decision),
                    explanation=str(explanation), status=status, closed=bool(closed),
                    notes=str(notes), priority=priority, source=str(source),
                    amount=float(amount), tx_type=str(tx_type), created_at=stamp, updated_at=stamp)

    def update(self, value: str | int, *, status: str | None = None, notes: str | None = None,
               priority: str | None = None, closed: bool | None = None) -> Case:
        """Change status, notes, priority and/or the closed state. Only real changes touch updated_at."""
        case = self.get(value)
        if case is None:
            raise KeyError(f"Case {value!r} does not exist.")
        new_status = _check_status(status) if status is not None else case.status
        new_priority = _check_priority(priority) if priority is not None else case.priority
        new_notes = case.notes if notes is None else str(notes)
        new_closed = case.closed if closed is None else bool(closed)

        changed = (new_status, new_priority, new_notes, new_closed) != \
                  (case.status, case.priority, case.notes, case.closed)
        updated = _now() if changed else case.updated_at
        conn = self._connect()
        try:
            conn.execute("UPDATE cases SET status = ?, notes = ?, priority = ?, closed = ?,"
                         " updated_at = ? WHERE id = ?",
                         (new_status, new_notes, new_priority, 1 if new_closed else 0,
                          updated, case.number))
            conn.commit()
        finally:
            if self._memory is None:
                conn.close()
        return replace(case, status=new_status, notes=new_notes, priority=new_priority,
                       closed=new_closed, updated_at=updated)

    def set_status(self, value: str | int, status: str) -> Case:
        """Shortcut for moving a case's resolution."""
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


def auto_open(store: "CaseStore", rows, min_risk: float, limit: int = 0,
              source: str = "auto") -> int:
    """Open a case for every alert at or above `min_risk`; return how many were made.

    This is the automatic version of the analyst clicking "Open investigation
    case": high-risk alerts are too easy to lose when the feed keeps moving, so
    the ones that reach the threshold are copied into SQLite immediately.

    `rows` is any iterable of objects with `id`, `risk`, `decision`, `reason`,
    `amount` and `type` (a DataFrame's `itertuples()` fits). Approved
    transactions are skipped, and so is any transaction that already has a case
    -- closed ones included, so reopening a decision never duplicates history.
    `limit` caps the batch: 0 means no cap.
    """
    made = 0
    for row in rows:
        if limit and made >= limit:
            break
        if str(getattr(row, "decision", "")) == "approve":
            continue
        risk = float(getattr(row, "risk", 0.0))
        if risk < float(min_risk):
            continue
        txn_id = int(getattr(row, "id"))
        if store.for_transaction(txn_id):            # a case already exists, open or closed
            continue
        store.create(txn_id, risk, str(getattr(row, "decision", "review")),
                     str(getattr(row, "reason", "")), source=source,
                     amount=float(getattr(row, "amount", 0.0) or 0.0),
                     tx_type=str(getattr(row, "type", "")))
        made += 1
    return made
