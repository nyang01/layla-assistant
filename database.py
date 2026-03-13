"""Database layer for multi-user Layla — PostgreSQL with encrypted token storage."""

import json
import os
import uuid
from datetime import datetime

import psycopg2
import psycopg2.extras
from cryptography.fernet import Fernet

# --- Encryption ---

_fernet = None


def _get_fernet() -> Fernet:
    """Get Fernet cipher for encrypting/decrypting OAuth refresh tokens."""
    global _fernet
    if _fernet is None:
        key = os.getenv("TOKEN_ENCRYPTION_KEY")
        if not key:
            raise RuntimeError("TOKEN_ENCRYPTION_KEY not set. Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def _encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# --- Connection ---

def _get_conn():
    """Get a PostgreSQL connection from DATABASE_URL."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set.")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# --- Schema ---

def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT DEFAULT '',
                    google_refresh_token TEXT NOT NULL,
                    google_access_token TEXT,
                    google_token_expiry TEXT,
                    api_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY REFERENCES users(id),
                    contacts TEXT DEFAULT '{}',
                    facts TEXT DEFAULT '[]',
                    last_session_timestamp TEXT
                );
            """)
        conn.commit()
        print("[Database] Tables ready")
    finally:
        conn.close()


# --- User CRUD ---

def get_user_by_api_token(api_token: str) -> dict | None:
    """Look up a user by their API token (used for auth middleware)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name, api_token FROM users WHERE api_token = %s", (api_token,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_google_id(google_id: str) -> dict | None:
    """Look up a user by their Google user ID."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name, api_token FROM users WHERE id = %s", (google_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def create_user(google_id: str, email: str, name: str, refresh_token: str) -> dict:
    """Create a new user and their memory record. Returns the user dict with api_token."""
    api_token = uuid.uuid4().hex
    encrypted_refresh = _encrypt(refresh_token)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO users (id, email, name, google_refresh_token, api_token)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       email = EXCLUDED.email,
                       name = EXCLUDED.name,
                       google_refresh_token = EXCLUDED.google_refresh_token
                   RETURNING id, email, name, api_token""",
                (google_id, email, name, encrypted_refresh, api_token),
            )
            user = dict(cur.fetchone())

            # Create memory record
            cur.execute(
                """INSERT INTO user_memory (user_id) VALUES (%s) ON CONFLICT DO NOTHING""",
                (google_id,),
            )
        conn.commit()
        return user
    finally:
        conn.close()


def update_user_tokens(google_id: str, access_token: str, expiry: str, refresh_token: str | None = None) -> None:
    """Update a user's OAuth tokens after a refresh."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if refresh_token:
                encrypted_refresh = _encrypt(refresh_token)
                cur.execute(
                    """UPDATE users SET google_access_token = %s, google_token_expiry = %s,
                       google_refresh_token = %s WHERE id = %s""",
                    (access_token, expiry, encrypted_refresh, google_id),
                )
            else:
                cur.execute(
                    """UPDATE users SET google_access_token = %s, google_token_expiry = %s WHERE id = %s""",
                    (access_token, expiry, google_id),
                )
        conn.commit()
    finally:
        conn.close()


def get_user_credentials_data(user_id: str) -> dict | None:
    """Get OAuth token data for building a Credentials object."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT google_refresh_token, google_access_token, google_token_expiry
                   FROM users WHERE id = %s""",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "refresh_token": _decrypt(row["google_refresh_token"]),
                "access_token": row["google_access_token"],
                "expiry": row["google_token_expiry"],
            }
    finally:
        conn.close()


# --- Memory CRUD ---

def get_user_memory(user_id: str) -> dict:
    """Load a user's memory (contacts, facts, last_session_timestamp) merged with user info."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT u.email, u.name, m.contacts, m.facts, m.last_session_timestamp
                   FROM users u LEFT JOIN user_memory m ON u.id = m.user_id
                   WHERE u.id = %s""",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return {"user_email": "", "user_name": "", "contacts": {}, "facts": [], "last_session_timestamp": None}

            contacts = json.loads(row["contacts"]) if row["contacts"] else {}
            facts = json.loads(row["facts"]) if row["facts"] else []

            return {
                "user_email": row["email"] or "",
                "user_name": row["name"] or "",
                "contacts": contacts,
                "facts": facts,
                "last_session_timestamp": row["last_session_timestamp"],
            }
    finally:
        conn.close()


def save_user_memory(user_id: str, memory: dict) -> None:
    """Save a user's memory (contacts, facts, last_session_timestamp)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO user_memory (user_id, contacts, facts, last_session_timestamp)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (user_id) DO UPDATE SET
                       contacts = EXCLUDED.contacts,
                       facts = EXCLUDED.facts,
                       last_session_timestamp = EXCLUDED.last_session_timestamp""",
                (
                    user_id,
                    json.dumps(memory.get("contacts", {}), ensure_ascii=False),
                    json.dumps(memory.get("facts", []), ensure_ascii=False),
                    memory.get("last_session_timestamp"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
