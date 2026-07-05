import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from backend.retail_config import RETENTION_DAYS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCREENSHOT_DIR = DATA_DIR / "screenshots"
DB_PATH = DATA_DIR / "jewelguard_events.db"

DATA_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cursor, table, col, typedef):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
    except sqlite3.OperationalError:
        pass


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            risk_description TEXT NOT NULL,
            mask_status TEXT,
            mask_confidence REAL,
            face_covering_detected INTEGER,
            people_near_case INTEGER,
            wrist_near_case INTEGER,
            motion_level TEXT,
            motion_score REAL,
            repeated_high_motion INTEGER,
            loitering_seconds INTEGER,
            screenshot_path TEXT
        )
        """
    )

    for col, typedef in (
        ("alert_zone", "TEXT DEFAULT 'general'"),
        ("status", "TEXT DEFAULT 'open'"),
        ("staff_note", "TEXT"),
        ("acknowledged_at", "TEXT"),
        ("resolved_at", "TEXT"),
        ("updated_at", "TEXT"),
    ):
        _ensure_column(cursor, "incidents", col, typedef)

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS staff_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            target_id TEXT,
            staff_name TEXT,
            detail TEXT
        )
        """
    )

    conn.commit()
    conn.close()

    if RETENTION_DAYS > 0:
        purge_old_incidents(RETENTION_DAYS)


def purge_old_incidents(days: int):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE timestamp < ?", (cutoff,))
    conn.commit()
    conn.close()


def clear_all_incidents(*, delete_screenshots=True, clear_staff_actions=False):
    """Remove all incidents (and optionally screenshots) for a fresh test run."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents")
    incident_count = cursor.rowcount
    staff_count = 0
    if clear_staff_actions:
        cursor.execute("DELETE FROM staff_actions")
        staff_count = cursor.rowcount
    conn.commit()
    conn.close()

    screenshot_count = 0
    if delete_screenshots:
        for path in SCREENSHOT_DIR.glob("*.jpg"):
            path.unlink(missing_ok=True)
            screenshot_count += 1

    return {
        "incidentsDeleted": incident_count,
        "screenshotsDeleted": screenshot_count,
        "staffActionsDeleted": staff_count,
    }


def insert_incident(
    person_id,
    risk_score,
    risk_level,
    risk_description,
    mask_status,
    mask_confidence,
    face_covering_detected,
    people_near_case,
    wrist_near_case,
    motion_level,
    motion_score,
    repeated_high_motion,
    loitering_seconds,
    screenshot_path,
    alert_zone="general",
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO incidents (
            person_id,
            timestamp,
            risk_score,
            risk_level,
            risk_description,
            mask_status,
            mask_confidence,
            face_covering_detected,
            people_near_case,
            wrist_near_case,
            motion_level,
            motion_score,
            repeated_high_motion,
            loitering_seconds,
            screenshot_path,
            alert_zone,
            status,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """,
        (
            person_id,
            now,
            risk_score,
            risk_level,
            risk_description,
            mask_status,
            mask_confidence,
            int(face_covering_detected),
            people_near_case,
            int(wrist_near_case),
            motion_level,
            motion_score,
            repeated_high_motion,
            loitering_seconds,
            screenshot_path,
            alert_zone,
            now,
        ),
    )

    incident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return incident_id


def get_incidents(limit=100, status=None):
    conn = get_connection()
    cursor = conn.cursor()

    if status:
        cursor.execute(
            """
            SELECT * FROM incidents
            WHERE status = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (status, limit),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_incident_by_id(incident_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def update_incident_status(incident_id, status, staff_note=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()

    fields = ["status = ?", "updated_at = ?"]
    values = [status, now]

    if status == "acknowledged":
        fields.append("acknowledged_at = ?")
        values.append(now)
    elif status == "resolved":
        fields.append("resolved_at = ?")
        values.append(now)

    if staff_note is not None:
        fields.append("staff_note = ?")
        values.append(staff_note)

    values.append(incident_id)
    cursor.execute(
        f"UPDATE incidents SET {', '.join(fields)} WHERE id = ?",
        values,
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def log_staff_action(action_type, target_id=None, staff_name="staff", detail=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO staff_actions (timestamp, action_type, target_id, staff_name, detail)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action_type,
            target_id,
            staff_name,
            detail,
        ),
    )
    conn.commit()
    conn.close()


def get_staff_actions(limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM staff_actions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def resolve_screenshot_path(path: str) -> Path | None:
    """Only allow files inside data/screenshots (blocks path traversal)."""
    if not path:
        return None
    try:
        candidate = Path(path).resolve()
        allowed = SCREENSHOT_DIR.resolve()
        if allowed in candidate.parents or candidate == allowed:
            return candidate if candidate.is_file() else None
    except OSError:
        return None
    return None
