import sqlite3
from datetime import datetime

conn = sqlite3.connect("traffic.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    cars INTEGER,
    bikes INTEGER,
    bus INTEGER,
    truck INTEGER,
    no_helmet INTEGER
)
""")

def insert_data(data):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO traffic_data (date, cars, bikes, bus, truck, no_helmet)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (date, data["cars"], data["bikes"], data["bus"], data["truck"], data["no_helmet"]))

    conn.commit()


def get_data():
    cursor.execute("SELECT * FROM traffic_data ORDER BY id DESC LIMIT 50")
    return cursor.fetchall()
