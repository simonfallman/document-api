import hashlib
import secrets
import sqlite3
from config import settings


def generate_key() -> tuple[str, str]:
    """Return (raw_key, key_hash). Only the hash is stored."""
    raw = "sk-" + secrets.token_urlsafe(32)
    return raw, hash_key(raw)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_api_key(name: str) -> str:
    """Create a new key, persist the hash, return the raw key (shown once)."""
    raw, key_hash = generate_key()
    con = sqlite3.connect(settings.db_path)
    con.execute(
        "INSERT INTO api_keys (key_hash, name) VALUES (?, ?)",
        (key_hash, name),
    )
    con.commit()
    con.close()
    return raw


def get_key_record(key_hash: str) -> dict | None:
    con = sqlite3.connect(settings.db_path)
    row = con.execute(
        "SELECT key_hash, name, is_active FROM api_keys WHERE key_hash = ?",
        (key_hash,),
    ).fetchone()
    con.close()
    if row:
        return {"key_hash": row[0], "name": row[1], "is_active": row[2]}
    return None


def touch_last_used(key_hash: str):
    con = sqlite3.connect(settings.db_path)
    con.execute(
        "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = ?",
        (key_hash,),
    )
    con.commit()
    con.close()
