import sqlite3
from pathlib import Path
from datetime import datetime

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

    conn.commit()
    conn.close()


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
):
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
            screenshot_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            person_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        ),
    )

    conn.commit()
    conn.close()


def get_incidents(limit=100):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM incidents
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

    cursor.execute(
        """
        SELECT *
        FROM incidents
        WHERE id = ?
        """,
        (incident_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)