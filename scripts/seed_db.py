"""
Seeds the SQLite database with quarterly revenue data.
"""

import os
import sqlite3

DB_PATH = os.getenv("DATABASE_PATH", "data/revenue.db")


def seed_database(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS quarterly_revenue (
        id INTEGER PRIMARY KEY,
        quarter VARCHAR NOT NULL,
        revenue REAL NOT NULL
    )
    """)
    c.execute("SELECT count(*) FROM quarterly_revenue")
    count = c.fetchone()[0]
    if count < 4:
        c.execute("DELETE FROM quarterly_revenue")
        rows = [
            (1, "Q1 2023", 3.8),
            (2, "Q2 2023", 4.1),
            (3, "Q3 2023", 4.2),
            (4, "Q4 2023", 4.5),
            (5, "Q1 2024", 4.7),
            (6, "Q2 2024", 5.0),
            (7, "Q3 2024", 5.3),
            (8, "Q4 2024", 5.6),
        ]
        c.executemany("INSERT INTO quarterly_revenue (id, quarter, revenue) VALUES (?, ?, ?)", rows)
        conn.commit()
        print(f"Seeded {len(rows)} rows into {db_path}.")
    else:
        print(f"Database at {db_path} already contains {count} rows.")
    conn.close()


if __name__ == "__main__":
    seed_database()
