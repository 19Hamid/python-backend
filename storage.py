import json
import os
import random
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def database():
    path = Path(os.getenv("BEAKSPEAK_DB_PATH", "data/beakspeak.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS stats (session_id TEXT PRIMARY KEY, questions_answered INTEGER NOT NULL DEFAULT 0, threat_level INTEGER NOT NULL DEFAULT 0, badges TEXT NOT NULL DEFAULT '[]')")
        connection.execute("CREATE TABLE IF NOT EXISTS quizzes (quiz_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, question TEXT NOT NULL, answered INTEGER NOT NULL DEFAULT 0)")
        connection.execute("CREATE INDEX IF NOT EXISTS quiz_session ON quizzes(session_id)")
        connection.execute("CREATE TABLE IF NOT EXISTS limits (key TEXT PRIMARY KEY, count INTEGER NOT NULL, expires REAL NOT NULL)")
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _stats(connection, session_id):
    connection.execute("INSERT OR IGNORE INTO stats(session_id) VALUES (?)", (session_id,))
    row = connection.execute("SELECT questions_answered, threat_level, badges FROM stats WHERE session_id=?", (session_id,)).fetchone()
    return {"questions_answered": row["questions_answered"], "threat_level": row["threat_level"], "badges": json.loads(row["badges"])}


def _record(connection, session_id, correct, badge):
    stats = _stats(connection, session_id)
    badges = stats["badges"] + ([badge] if correct and badge not in stats["badges"] else [])
    connection.execute("UPDATE stats SET questions_answered=questions_answered+1, threat_level=threat_level+?, badges=? WHERE session_id=?", (int(correct), json.dumps(badges), session_id))
    return _stats(connection, session_id)


def read_stats(session_id):
    with database() as connection:
        return _stats(connection, session_id)


def record_meal(session_id, correct):
    with database() as connection:
        return _record(connection, session_id, correct, "Carrion Expert")


def issue_quiz(session_id, questions):
    with database() as connection:
        rows = connection.execute("SELECT * FROM quizzes WHERE session_id=?", (session_id,)).fetchall()
        pending = next((row for row in rows if not row["answered"]), None)
        if pending:
            question = next(question for question in questions if question["question"] == pending["question"])
            return {"quiz_id": pending["quiz_id"], "question": question["question"], "options": question["options"]}
        remaining = [question for question in questions if question["question"] not in {row["question"] for row in rows}]
        if not remaining:
            connection.execute("DELETE FROM quizzes WHERE session_id=?", (session_id,))
            remaining = questions
        question = random.choice(remaining)
        quiz_id = str(uuid4())
        connection.execute("INSERT INTO quizzes(quiz_id,session_id,question) VALUES (?,?,?)", (quiz_id, session_id, question["question"]))
        return {"quiz_id": quiz_id, "question": question["question"], "options": question["options"]}


def record_quiz(session_id, quiz_id, question, answer):
    with database() as connection:
        if quiz_id:
            row = connection.execute("SELECT * FROM quizzes WHERE quiz_id=? AND session_id=?", (quiz_id, session_id)).fetchone()
        else:
            # Legacy clients can submit the issued question without a quiz ID.
            row = connection.execute("SELECT * FROM quizzes WHERE session_id=? AND question=?", (session_id, question["question"])).fetchone()
        if row is None or row["question"] != question["question"]:
            raise ValueError("Request this quiz question before answering it.")
        if row["answered"]:
            raise ValueError("This question has already been answered.")
        connection.execute("UPDATE quizzes SET answered=1 WHERE quiz_id=?", (row["quiz_id"],))
        return _record(connection, session_id, answer == question["answer"], question["badge"])


def rate_limit(ip_hash, timestamp):
    with database() as connection:
        connection.execute("DELETE FROM limits WHERE expires<=?", (timestamp,))
        windows = [("ip:" + ip_hash, 10, 60), ("global:minute", 120, 60), ("global:day", 1000, 86400)]
        for key, maximum, _ in windows:
            row = connection.execute("SELECT count,expires FROM limits WHERE key=?", (key,)).fetchone()
            if row and row["count"] >= maximum:
                return False, max(1, int(row["expires"] - timestamp + 1))
        if connection.execute("SELECT COUNT(*) FROM limits").fetchone()[0] >= 4096:
            return False, 60
        for key, _, seconds in windows:
            connection.execute("INSERT INTO limits(key,count,expires) VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1", (key, timestamp + seconds))
        return True, 0
