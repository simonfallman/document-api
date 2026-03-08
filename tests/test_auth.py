"""
Tests for API key generation, hashing, and validation dependency.
Uses real SQLite (isolated per test via conftest.isolated_settings).
"""
import pytest
from fastapi.testclient import TestClient


# ── Key generation and storage ────────────────────────────────────────────────

def test_create_api_key_returns_raw_key():
    from auth.keys import create_api_key
    raw = create_api_key("test")
    assert raw.startswith("sk-")
    assert len(raw) > 10


def test_created_key_is_retrievable_by_hash():
    from auth.keys import create_api_key, hash_key, get_key_record
    raw = create_api_key("retrieval-test")
    record = get_key_record(hash_key(raw))
    assert record is not None
    assert record["name"] == "retrieval-test"
    assert record["is_active"] == 1


def test_two_keys_have_different_hashes():
    from auth.keys import create_api_key, hash_key
    raw1 = create_api_key("key1")
    raw2 = create_api_key("key2")
    assert hash_key(raw1) != hash_key(raw2)


def test_unknown_key_returns_none():
    from auth.keys import get_key_record
    assert get_key_record("nonexistent-hash") is None


def test_hash_key_is_deterministic():
    from auth.keys import hash_key
    assert hash_key("sk-abc") == hash_key("sk-abc")


# ── Validation via HTTP ───────────────────────────────────────────────────────

def test_valid_key_is_accepted(client):
    # client fixture overrides auth — just verify the health endpoint works
    resp = client.get("/health")
    assert resp.status_code == 200


def test_missing_key_returns_422():
    """Without the dependency override, a missing X-API-Key header returns 422."""
    from main import app
    with TestClient(app) as c:
        resp = c.get("/conversations")
    assert resp.status_code == 422


def test_invalid_key_returns_401():
    from main import app
    with TestClient(app) as c:
        resp = c.get("/conversations", headers={"X-API-Key": "sk-invalid"})
    assert resp.status_code == 401


def test_valid_real_key_returns_200():
    from main import app
    from auth.keys import create_api_key
    raw = create_api_key("http-test")
    with TestClient(app) as c:
        resp = c.get("/conversations", headers={"X-API-Key": raw})
    assert resp.status_code == 200


# ── Admin key creation endpoint ───────────────────────────────────────────────

def test_admin_create_key_requires_master_key(client):
    resp = client.post("/admin/keys?name=new", headers={"X-Master-Key": "wrong"})
    assert resp.status_code == 401


def test_admin_create_key_with_correct_master_key():
    from main import app
    from config import settings
    with TestClient(app) as c:
        resp = c.post(
            "/admin/keys?name=mykey",
            headers={"X-Master-Key": settings.api_master_key},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["key"].startswith("sk-")
    assert data["name"] == "mykey"
