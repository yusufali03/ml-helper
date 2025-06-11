# logger.py

import json
from datetime import datetime
import sqlite3
from database import DB_PATH, init_db

def log_task(task: str, params: dict, status: str, details: str = ""):
    """
    Inserts a log row into the SQLite database.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (timestamp, task, params, status, details) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), task, json.dumps(params), status, details),
    )
    conn.commit()
    conn.close()
