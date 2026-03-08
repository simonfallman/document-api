"""
Tests for POST /documents/upload.
Bedrock calls are mocked so no AWS credentials are needed.
"""
import io
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _txt_upload(client, content: str = "Hello world. " * 50, filename: str = "test.txt"):
    """Upload a plain-text file and return the response."""
    return client.post(
        "/documents/upload",
        files={"file": (filename, io.BytesIO(content.encode()), "text/plain")},
    )


# ── Upload happy path ─────────────────────────────────────────────────────────

def test_upload_txt_returns_document_id(client, mock_embeddings):
    with patch("routers.documents.build_vectorstore", return_value=5):
        resp = _txt_upload(client)
    assert resp.status_code == 200
    data = resp.json()
    assert "document_id" in data
    assert data["filename"] == "test.txt"
    assert data["chunks"] == 5


def test_upload_same_file_twice_is_idempotent(client, mock_embeddings):
    content = "Same content. " * 50
    with patch("routers.documents.build_vectorstore", return_value=3) as mock_vs:
        r1 = _txt_upload(client, content=content, filename="doc.txt")
        r2 = _txt_upload(client, content=content, filename="doc.txt")

    assert r1.status_code == r2.status_code == 200
    assert r1.json()["document_id"] == r2.json()["document_id"]
    # build_vectorstore should only be called once (second upload is cached)
    mock_vs.assert_called_once()


def test_upload_different_files_produce_different_ids(client, mock_embeddings):
    with patch("routers.documents.build_vectorstore", return_value=2):
        r1 = _txt_upload(client, content="File one content. " * 40, filename="a.txt")
        r2 = _txt_upload(client, content="File two content. " * 40, filename="b.txt")
    assert r1.json()["document_id"] != r2.json()["document_id"]


# ── Validation errors ─────────────────────────────────────────────────────────

def test_upload_unsupported_type_returns_400(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"data"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_file_too_large_returns_400(client):
    big = b"x" * (6 * 1024 * 1024)  # 6MB > 5MB limit
    resp = client.post(
        "/documents/upload",
        files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
    )
    assert resp.status_code == 400
    assert "large" in resp.json()["detail"].lower()


# ── document_id format ────────────────────────────────────────────────────────

def test_document_id_is_md5_hex(client, mock_embeddings):
    with patch("routers.documents.build_vectorstore", return_value=1):
        resp = _txt_upload(client)
    doc_id = resp.json()["document_id"]
    assert len(doc_id) == 32
    assert all(c in "0123456789abcdef" for c in doc_id)
