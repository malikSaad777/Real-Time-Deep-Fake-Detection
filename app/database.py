# app/database.py
import sqlite3

def init_db():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Create logs table (if not already created)
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_name TEXT,
                        label TEXT,
                        confidence REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')

    # Create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()
