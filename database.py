"""
database.py
===========
Handles all SQLite database operations.
Every detection is saved permanently — survives server restarts.
Provides search, filter, pagination for the History page.
"""

import json
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(LOG_DIR, "ids_database.db")


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = _connect()
    c    = conn.cursor()
    c.execute("PRAGMA journal_mode = WAL")

    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        INTEGER,
            record_id         INTEGER,
            timestamp         TEXT,
            label             TEXT,
            category          TEXT,
            is_attack         INTEGER,
            protocol          TEXT,
            service           TEXT,
            src_bytes         INTEGER,
            duration          INTEGER,
            confidence        REAL,
            shap_top          TEXT,
            actual_label      TEXT,
            actual_category   TEXT,
            validation_status TEXT,
            severity          TEXT
        )
    """)

    # Migrate existing databases — add columns if missing
    existing = {row[1] for row in c.execute("PRAGMA table_info(detections)").fetchall()}
    migrations = {
        "session_id":        "ALTER TABLE detections ADD COLUMN session_id INTEGER",
        "actual_label":      "ALTER TABLE detections ADD COLUMN actual_label TEXT",
        "actual_category":   "ALTER TABLE detections ADD COLUMN actual_category TEXT",
        "validation_status": "ALTER TABLE detections ADD COLUMN validation_status TEXT",
        "severity":          "ALTER TABLE detections ADD COLUMN severity TEXT",
    }
    for col, sql in migrations.items():
        if col not in existing:
            try:
                c.execute(sql)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            existing.add(col)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time   TEXT,
            total      INTEGER DEFAULT 0,
            attacks    INTEGER DEFAULT 0,
            normal     INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        DELETE FROM detections
        WHERE session_id IS NULL
          AND id NOT IN (
              SELECT MIN(id)
              FROM detections
              WHERE session_id IS NULL
              GROUP BY record_id
          )
    """)

    legacy_count = c.execute(
        "SELECT COUNT(*) FROM detections WHERE session_id IS NULL"
    ).fetchone()[0]
    if legacy_count:
        row = c.execute("""
            SELECT id FROM sessions
            WHERE start_time = 'legacy-import'
            LIMIT 1
        """).fetchone()
        if row:
            legacy_session_id = row[0]
        else:
            c.execute("""
                INSERT INTO sessions (start_time, end_time)
                VALUES ('legacy-import', 'legacy-import')
            """)
            legacy_session_id = c.lastrowid

        c.execute(
            "UPDATE detections SET session_id = ? WHERE session_id IS NULL",
            (legacy_session_id,)
        )

        total, attacks = c.execute("""
            SELECT COUNT(*), COALESCE(SUM(is_attack), 0)
            FROM detections
            WHERE session_id = ?
        """, (legacy_session_id,)).fetchone()
        c.execute("""
            UPDATE sessions
            SET total = ?, attacks = ?, normal = ?
            WHERE id = ?
        """, (
            int(total), int(attacks), int(total - attacks), legacy_session_id
        ))

    c.execute("CREATE INDEX IF NOT EXISTS idx_category ON detections(category)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_is_attack ON detections(is_attack)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_session ON detections(session_id)")
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_record
        ON detections(session_id, record_id)
        WHERE session_id IS NOT NULL
    """)

    conn.commit()
    conn.close()


def start_session():
    """Create a new dashboard stream session and return its id."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (start_time) VALUES (?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    )
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def finish_session(session_id, total=0, attacks=0, normal=0):
    """Mark a session complete and store its final counters."""
    if session_id is None:
        return
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        UPDATE sessions
        SET end_time = ?, total = ?, attacks = ?, normal = ?
        WHERE id = ?
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        int(total), int(attacks), int(normal), int(session_id)
    ))
    conn.commit()
    conn.close()


def get_latest_session_id():
    """Return the newest session id, or None if no sessions exist."""
    conn = _connect()
    c = conn.cursor()
    row = c.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row[0] if row else None


def _session_filter(session_id, conditions, params):
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(int(session_id))


def insert_detection(alert: dict, session_id=None):
    """Save one detection to the database."""
    conn = _connect()
    c    = conn.cursor()

    c.execute("""
        INSERT OR IGNORE INTO detections
            (session_id, record_id, timestamp, label, category,
             is_attack, protocol, service, src_bytes,
             duration, confidence, shap_top,
             actual_label, actual_category, validation_status, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        alert.get("id",               0),
        alert.get("timestamp",        ""),
        alert.get("label",            ""),
        alert.get("category",         ""),
        1 if alert.get("is_attack") else 0,
        alert.get("protocol",         ""),
        alert.get("service",          ""),
        alert.get("src_bytes",        0),
        alert.get("duration",         0),
        round(float(alert.get("confidence", 0.0)), 4),
        json.dumps(alert.get("shap_top", [])),
        alert.get("actual_label",     ""),
        alert.get("actual_category",  ""),
        alert.get("validation_status",""),
        alert.get("severity",         ""),
    ))

    conn.commit()
    conn.close()


def get_detections(limit=50, offset=0, category=None, search=None,
                   attacks_only=False, session_id=None):
    """Paginated detections with optional filters."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()

    conditions, params = [], []
    _session_filter(session_id, conditions, params)

    if attacks_only:
        conditions.append("is_attack = 1")
    if category and category.lower() != "all":
        conditions.append("category = ?")
        params.append(category)
    if search and search.strip():
        s = f"%{search.strip()}%"
        conditions.append("(label LIKE ? OR service LIKE ? OR protocol LIKE ? OR category LIKE ?)")
        params.extend([s, s, s, s])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    c.execute(f"SELECT * FROM detections {where} ORDER BY id DESC LIMIT ? OFFSET ?",
              params + [limit, offset])

    rows = []
    for row in c.fetchall():
        d = dict(row)
        try:
            d["shap_top"] = json.loads(d.get("shap_top") or "[]")
        except Exception:
            d["shap_top"] = []
        rows.append(d)

    conn.close()
    return rows


def get_total_count(category=None, search=None, attacks_only=False,
                    session_id=None):
    """Total row count matching same filters."""
    conn = _connect()
    c    = conn.cursor()

    conditions, params = [], []
    _session_filter(session_id, conditions, params)

    if attacks_only:
        conditions.append("is_attack = 1")
    if category and category.lower() != "all":
        conditions.append("category = ?")
        params.append(category)
    if search and search.strip():
        s = f"%{search.strip()}%"
        conditions.append("(label LIKE ? OR service LIKE ? OR protocol LIKE ? OR category LIKE ?)")
        params.extend([s, s, s, s])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    c.execute(f"SELECT COUNT(*) FROM detections {where}", params)
    count = c.fetchone()[0]
    conn.close()
    return count


def get_db_stats(session_id=None):
    """Full statistics for Reports page and PDF."""
    conn = _connect()
    c    = conn.cursor()
    where = "WHERE session_id = ?" if session_id is not None else ""
    attack_where = (
        "WHERE is_attack = 1 AND session_id = ?"
        if session_id is not None else
        "WHERE is_attack = 1"
    )
    params = [int(session_id)] if session_id is not None else []

    c.execute(f"SELECT COUNT(*) FROM detections {where}", params)
    total = c.fetchone()[0]

    c.execute(f"SELECT COUNT(*) FROM detections {attack_where}", params)
    attacks = c.fetchone()[0]

    c.execute(f"""
        SELECT category, COUNT(*) FROM detections
        {attack_where}
        GROUP BY category ORDER BY COUNT(*) DESC
    """, params)
    by_category = {r[0]: r[1] for r in c.fetchall()}

    c.execute(f"""
        SELECT label, COUNT(*) FROM detections
        {attack_where}
        GROUP BY label ORDER BY COUNT(*) DESC LIMIT 10
    """, params)
    top_attacks = [{"label": r[0], "count": r[1]} for r in c.fetchall()]

    c.execute(f"""
        SELECT protocol, COUNT(*) FROM detections
        {where}
        GROUP BY protocol ORDER BY COUNT(*) DESC LIMIT 5
    """, params)
    by_protocol = [{"protocol": r[0], "count": r[1]} for r in c.fetchall()]

    c.execute(f"SELECT AVG(confidence) FROM detections {attack_where}", params)
    avg_row = c.fetchone()[0]
    avg_confidence = round(float(avg_row or 0), 1)

    # Validation status breakdown
    c.execute(f"""
        SELECT validation_status, COUNT(*) FROM detections
        {where}
        GROUP BY validation_status
    """, params)
    by_validation = {r[0]: r[1] for r in c.fetchall() if r[0]}

    # Severity breakdown
    c.execute(f"""
        SELECT severity, COUNT(*) FROM detections
        {attack_where}
        GROUP BY severity
    """, params)
    by_severity = {r[0]: r[1] for r in c.fetchall() if r[0]}

    # Confusion matrix totals from validation statuses
    tp = by_validation.get("Correct Detection", 0) + by_validation.get("Misclassification", 0)
    fp = by_validation.get("False Positive", 0)
    fn = by_validation.get("False Negative", 0)
    tn = by_validation.get("Correct Normal", 0)
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0.0

    conn.close()
    return {
        "total":          total,
        "attacks":        attacks,
        "normal":         total - attacks,
        "by_category":    by_category,
        "top_attacks":    top_attacks,
        "by_protocol":    by_protocol,
        "avg_confidence": avg_confidence,
        "by_validation":  by_validation,
        "by_severity":    by_severity,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
