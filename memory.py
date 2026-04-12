"""
memory.py — Persistent student memory using SQLite.

Stores per-student:
  - weak topics (question numbers + categories they struggle with)
  - past session scores
  - conversation history (last N turns for agent context)
  - study plan
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "student_memory.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id  TEXT PRIMARY KEY,
            created_at  TEXT DEFAULT (datetime('now')),
            name        TEXT DEFAULT 'Student'
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            exam_name   TEXT,
            score       INTEGER,
            total       INTEGER,
            percentage  REAL,
            played_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS weak_questions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            question_number TEXT,
            question_text   TEXT,
            q_type          TEXT,
            topic           TEXT,
            wrong_count     INTEGER DEFAULT 1,
            last_seen       TEXT DEFAULT (datetime('now')),
            UNIQUE(student_id, question_number)
        );

        CREATE TABLE IF NOT EXISTS conversation_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            role        TEXT,
            content     TEXT,
            tool_call_id TEXT,
            tool_name   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS study_plan (
            student_id  TEXT PRIMARY KEY,
            plan        TEXT,
            updated_at  TEXT DEFAULT (datetime('now'))
        );
        """)


# ── Student ───────────────────────────────────────────────
def ensure_student(student_id, name="Student"):
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO students(student_id, name) VALUES(?,?)",
            (student_id, name)
        )


# ── Session history ───────────────────────────────────────
def save_session(student_id, exam_name, score, total, percentage):
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions(student_id,exam_name,score,total,percentage) VALUES(?,?,?,?,?)",
            (student_id, exam_name, score, total, percentage)
        )


def get_sessions(student_id, limit=5):
    with _conn() as c:
        rows = c.execute(
            "SELECT exam_name,score,total,percentage,played_at FROM sessions "
            "WHERE student_id=? ORDER BY played_at DESC LIMIT ?",
            (student_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Weak question tracking ────────────────────────────────
def record_wrong(student_id, question_number, question_text, q_type, topic=""):
    with _conn() as c:
        c.execute("""
            INSERT INTO weak_questions(student_id,question_number,question_text,q_type,topic,wrong_count,last_seen)
            VALUES(?,?,?,?,?,1,datetime('now'))
            ON CONFLICT(student_id,question_number) DO UPDATE SET
                wrong_count = wrong_count + 1,
                last_seen   = datetime('now'),
                question_text = excluded.question_text
        """, (student_id, question_number, question_text, q_type, topic))


def record_correct(student_id, question_number):
    """Reduce wrong_count by 1 when student gets it right (min 0)."""
    with _conn() as c:
        c.execute("""
            UPDATE weak_questions
            SET wrong_count = MAX(0, wrong_count - 1)
            WHERE student_id=? AND question_number=?
        """, (student_id, question_number))


def get_weak_topics(student_id, limit=10):
    with _conn() as c:
        rows = c.execute("""
            SELECT question_number, question_text, q_type, topic, wrong_count, last_seen
            FROM weak_questions
            WHERE student_id=? AND wrong_count > 0
            ORDER BY wrong_count DESC, last_seen DESC
            LIMIT ?
        """, (student_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_weak_summary(student_id):
    """Returns a short text summary of weak areas for agent context."""
    weak = get_weak_topics(student_id)
    if not weak:
        return "No weak areas recorded yet."
    lines = [f"Q{w['question_number']} ({w['q_type']}): wrong {w['wrong_count']}x" for w in weak]
    return "Weak questions: " + ", ".join(lines)


# ── Conversation history ──────────────────────────────────
def append_message(student_id, role, content, tool_call_id=None, tool_name=None):
    with _conn() as c:
        c.execute(
            "INSERT INTO conversation_history(student_id,role,content,tool_call_id,tool_name) VALUES(?,?,?,?,?)",
            (student_id, role,
             content if isinstance(content, str) else json.dumps(content),
             tool_call_id, tool_name)
        )


def get_history(student_id, limit=20):
    """
    Returns last N messages as a list of dicts suitable for the Groq API.
    Handles regular messages and tool call/result turns.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT role,content,tool_call_id,tool_name FROM conversation_history "
            "WHERE student_id=? ORDER BY id DESC LIMIT ?",
            (student_id, limit)
        ).fetchall()

    messages = []
    for r in reversed(rows):
        msg = {"role": r["role"]}
        try:
            msg["content"] = json.loads(r["content"])
        except Exception:
            msg["content"] = r["content"]

        if r["tool_call_id"]:
            msg["tool_call_id"] = r["tool_call_id"]
        if r["tool_name"]:
            msg["name"] = r["tool_name"]
        messages.append(msg)

    return messages


def clear_history(student_id):
    with _conn() as c:
        c.execute("DELETE FROM conversation_history WHERE student_id=?", (student_id,))


# ── Study plan ────────────────────────────────────────────
def save_study_plan(student_id, plan_text):
    with _conn() as c:
        c.execute("""
            INSERT INTO study_plan(student_id, plan, updated_at)
            VALUES(?,?,datetime('now'))
            ON CONFLICT(student_id) DO UPDATE SET plan=excluded.plan, updated_at=datetime('now')
        """, (student_id, plan_text))


def get_study_plan(student_id):
    with _conn() as c:
        row = c.execute(
            "SELECT plan, updated_at FROM study_plan WHERE student_id=?",
            (student_id,)
        ).fetchone()
    return dict(row) if row else None


# Init on import
init_db()