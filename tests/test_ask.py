"""
Tests for POST /ask and GET/DELETE /conversations.
The RAG chain is mocked so no AWS credentials are needed.
"""
import json
import io
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from services.storage import file_hash, register_document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_doc(isolated_settings):
    """Insert a document record directly into the test DB. Returns doc_id."""
    doc_id = file_hash(b"test document content for rag tests")
    register_document(doc_id, "test.txt", 5)
    return doc_id


def _mock_chain(answer: str = "The answer is 42.", docs: list = None):
    """Return a mock chain that returns a canned answer."""
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": answer,
        "context": docs or [],
    }
    return chain


# ── POST /ask happy path ──────────────────────────────────────────────────────

def test_ask_returns_answer(client, seeded_doc):
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain()):
        resp = client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "What is the meaning of life?",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer is 42."
    assert "conversation_id" in data


def test_ask_with_sources(client, seeded_doc):
    doc = Document(
        page_content="Relevant chunk of text.",
        metadata={"collection_hash": seeded_doc, "page": 1, "document_name": "test.txt"},
    )
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain(docs=[doc])):
        resp = client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "Tell me about section 2.",
        })
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert len(sources) == 1
    assert sources[0]["excerpt"] == "Relevant chunk of text."
    assert sources[0]["page"] == 2  # 0-indexed page + 1


def test_ask_generates_conversation_id_when_not_provided(client, seeded_doc):
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain()):
        resp = client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "Hello?",
        })
    cid = resp.json()["conversation_id"]
    assert len(cid) == 36  # UUID4 format


def test_ask_reuses_provided_conversation_id(client, seeded_doc):
    cid = "my-existing-conversation-id"
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain()):
        resp = client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "Follow-up question.",
            "conversation_id": cid,
        })
    assert resp.json()["conversation_id"] == cid


# ── POST /ask validation errors ───────────────────────────────────────────────

def test_ask_empty_document_ids_returns_400(client):
    resp = client.post("/ask", json={"document_ids": [], "question": "hello"})
    assert resp.status_code == 400


def test_ask_unknown_document_id_returns_404(client):
    resp = client.post("/ask", json={
        "document_ids": ["nonexistent-md5-hash-00000000"],
        "question": "hello",
    })
    assert resp.status_code == 404


# ── GET /conversations ────────────────────────────────────────────────────────

def test_list_conversations_initially_empty(client):
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_conversations_after_ask(client, seeded_doc):
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain()):
        client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "Test question for conversation listing.",
        })
    resp = client.get("/conversations")
    assert resp.status_code == 200
    convs = resp.json()
    assert len(convs) == 1
    assert seeded_doc in convs[0]["document_ids"]


# ── DELETE /conversations/{id} ────────────────────────────────────────────────

def test_delete_conversation(client, seeded_doc):
    with patch("routers.ask.get_vectorstores", return_value=[MagicMock()]), \
         patch("routers.ask.build_chain", return_value=_mock_chain()):
        ask_resp = client.post("/ask", json={
            "document_ids": [seeded_doc],
            "question": "Delete me.",
        })
    cid = ask_resp.json()["conversation_id"]

    del_resp = client.delete(f"/conversations/{cid}")
    assert del_resp.status_code == 204

    resp = client.get("/conversations")
    assert all(c["id"] != cid for c in resp.json())


def test_delete_nonexistent_conversation_returns_404(client):
    resp = client.delete("/conversations/does-not-exist")
    assert resp.status_code == 404
