import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "app.db")


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                topic TEXT NOT NULL,
                deadline TEXT NOT NULL,
                share_code TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                strengths TEXT,
                priority TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                due_date TEXT,
                review_criteria TEXT,
                done INTEGER DEFAULT 0,
                assignee_id INTEGER,
                ai_recommended_assignee TEXT,
                ai_progress_estimate INTEGER,
                ai_reasoning TEXT,
                sort_order INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                FOREIGN KEY (assignee_id) REFERENCES members(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                uploader_name TEXT,
                uploaded_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            """
        )

        # ---- 마이그레이션: 이미 배포되어 있던 DB에도 AI 피드백 컬럼을 추가한다 ----
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_files)").fetchall()
        }
        if "ai_feedback" not in existing_cols:
            conn.execute("ALTER TABLE task_files ADD COLUMN ai_feedback TEXT")
        if "ai_feedback_at" not in existing_cols:
            conn.execute("ALTER TABLE task_files ADD COLUMN ai_feedback_at TEXT")

        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
