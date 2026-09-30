
import sqlite3
import hashlib
import secrets
import re
from pathlib import Path
from contextlib import contextmanager

# Database location
DB_PATH = Path(__file__).resolve().parent / "users.db"


# Connect to database and always close the connection
@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Create users table
def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)


# Hash password securely
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000
    ).hex()

    return hashed, salt


# Register a new user
def register_user(full_name, email, username, password):
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()
    username = (username or "").strip()
    password = password or ""

    # Required fields
    if not full_name or not email or not username or not password:
        return False, "Please fill in every field."

    # Basic email validation
    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(email_pattern, email):
        return False, "Enter a valid email address."

    # Username validation
    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if not re.match(r"^[A-Za-z0-9_.-]+$", username):
        return False, (
            "Username can contain letters, numbers, "
            "dots, underscores, and hyphens only."
        )

    # Password validation
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    # Hash password before storing
    password_hash, salt = hash_password(password)

    try:
        with connect() as conn:
            conn.execute("""
                INSERT INTO users (
                    full_name,
                    email,
                    username,
                    password_hash,
                    salt
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                full_name,
                email,
                username,
                password_hash,
                salt
            ))

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."

    except sqlite3.Error:
        return False, "Database error. Please try again."


# Login using username or email
def login_user(username_or_email, password):
    identifier = (username_or_email or "").strip()
    password = password or ""

    if not identifier or not password:
        return None

    try:
        with connect() as conn:
            user = conn.execute("""
                SELECT *
                FROM users
                WHERE username = ? OR email = ?
            """, (identifier, identifier)).fetchone()

    except sqlite3.Error:
        return None

    if user is None:
        return None

    # Verify entered password
    entered_hash, _ = hash_password(
        password,
        user["salt"]
    )

    if secrets.compare_digest(
        entered_hash,
        user["password_hash"]
    ):
        return {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "username": user["username"]
        }

    return None


# Initialize database when this module is imported
init_db()