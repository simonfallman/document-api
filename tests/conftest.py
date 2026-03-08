"""
Test configuration and shared fixtures.

Strategy:
- Patch settings.db_path / chroma_dir / documents_dir to tmp_path for isolation.
- Override validate_api_key and check_rate_limit dependencies to skip real auth in
  endpoint tests. Individual auth tests use a real DB and real key validation.
- Mock BedrockEmbeddings and ChatBedrock so tests never hit AWS.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Settings isolation ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Redirect all data paths to a temp directory for every test."""
    from config import settings
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "documents_dir", str(tmp_path / "docs"))
    # Initialise fresh DB at the new path
    from database import init_db
    init_db()


# ── API client with auth bypassed ─────────────────────────────────────────────

TEST_KEY_HASH = "test-key-hash-abc123"


@pytest.fixture
def client():
    """TestClient with auth dependencies overridden so tests don't need real keys."""
    from main import app
    from auth.dependencies import validate_api_key
    from middleware.rate_limit import check_rate_limit

    app.dependency_overrides[validate_api_key] = lambda: TEST_KEY_HASH
    app.dependency_overrides[check_rate_limit] = lambda: TEST_KEY_HASH

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Bedrock mocks ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    """Mock BedrockEmbeddings to return deterministic fake vectors."""
    with patch("services.embeddings.get_embeddings") as mock:
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = embeddings
        yield embeddings


@pytest.fixture
def mock_llm():
    """Mock ChatBedrock to return a canned answer without hitting AWS."""
    with patch("services.rag._get_llm") as mock:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Mocked LLM answer")
        mock.return_value = llm
        yield llm
