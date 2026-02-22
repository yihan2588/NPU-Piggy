import sqlite3
from models import Receipt

DB_NAME = "receipts.db"

def init_db():
    """Create the receipts table if it doesn't exist"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receiver TEXT,
            date TEXT,
            total_amount REAL,
            currency TEXT,
            category TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized.")

def insert_receipt(receipt: Receipt):
    """Insert a parsed receipt into the database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO receipts (receiver, date, total_amount, currency, category, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        receipt.receiver,
        receipt.date,
        receipt.total_amount,
        receipt.currency,
        receipt.category,
        receipt.confidence
    ))
    
    conn.commit()
    conn.close()
    print(f"Receipt saved: {receipt.receiver} - {receipt.total_amount} {receipt.currency}")

def get_all_receipts():
    """Fetch all receipts from the database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM receipts")
    rows = cursor.fetchall()
    conn.close()
    return rows