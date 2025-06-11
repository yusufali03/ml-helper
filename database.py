# database.py

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "logs", "log.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            task TEXT,
            params TEXT,
            status TEXT,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()
