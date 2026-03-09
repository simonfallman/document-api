"""
Unit tests for RAG pipeline helpers.
Ported and extended from the chatbot's test_pipeline.py.
No AWS calls needed — only tests pure Python logic.
"""
import os
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unittest.mock import MagicMock

from services.storage import file_hash, SUPPORTED_SUFFIXES
from services.rag import SUMMARIZE_TRIGGERS, FAQ_TRIGGERS, multi_retrieve, format_docs, _get_llm


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunking_splits_long_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="word " * 100)]
    assert len(splitter.split_documents(docs)) > 1


def test_chunking_short_text_stays_single_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content="Short document.")]
    assert len(splitter.split_documents(docs)) == 1


def test_chunk_size_respected():
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content="a" * 200)]
    for chunk in splitter.split_documents(docs):
        assert len(chunk.page_content) <= 50


# ── File hashing ──────────────────────────────────────────────────────────────

def test_file_hash_is_deterministic():
    assert file_hash(b"hello world") == file_hash(b"hello world")


def test_different_files_have_different_hashes():
    assert file_hash(b"file one") != file_hash(b"file two")


def test_file_hash_is_md5_length():
    assert len(file_hash(b"test")) == 32


# ── Supported file types ──────────────────────────────────────────────────────

def test_supported_suffixes_includes_pdf_txt_docx():
    assert ".pdf" in SUPPORTED_SUFFIXES
    assert ".txt" in SUPPORTED_SUFFIXES
    assert ".docx" in SUPPORTED_SUFFIXES


# ── Intent detection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [
    "please summarize this document",
    "Can you give me a summary?",
    "TL;DR",
    "what is this document about",
    "give me an overview",
])
def test_summarize_triggers_match(query):
    assert SUMMARIZE_TRIGGERS.search(query)


@pytest.mark.parametrize("query", [
    "generate FAQ for this",
    "What are the key questions?",
    "quiz me on this content",
    "frequently asked questions",
])
def test_faq_triggers_match(query):
    assert FAQ_TRIGGERS.search(query)


def test_no_trigger_on_regular_question():
    query = "what is the main topic of chapter 2?"
    assert not SUMMARIZE_TRIGGERS.search(query)
    assert not FAQ_TRIGGERS.search(query)


# ── multi_retrieve ────────────────────────────────────────────────────────────

def _make_vectorstore(docs_with_scores: list[tuple[Document, float]]):
    vs = MagicMock()
    vs.similarity_search_with_relevance_scores.return_value = docs_with_scores
    return vs


def test_multi_retrieve_merges_and_deduplicates():
    doc_a = Document(page_content="chunk A")
    doc_b = Document(page_content="chunk B")
    doc_dup = Document(page_content="chunk A")  # duplicate of doc_a

    vs1 = _make_vectorstore([(doc_a, 0.9), (doc_b, 0.7)])
    vs2 = _make_vectorstore([(doc_dup, 0.85)])

    results = multi_retrieve([vs1, vs2], "test query", k=6)
    contents = [d.page_content for d in results]

    # chunk A should appear only once despite being in both stores
    assert contents.count("chunk A") == 1
    assert "chunk B" in contents


def test_multi_retrieve_respects_k():
    docs = [(Document(page_content=f"chunk {i}"), 0.9 - i * 0.1) for i in range(10)]
    vs = _make_vectorstore(docs)
    results = multi_retrieve([vs], "query", k=3)
    assert len(results) == 3


def test_multi_retrieve_orders_by_score():
    low = Document(page_content="low score")
    high = Document(page_content="high score")
    vs = _make_vectorstore([(low, 0.3), (high, 0.9)])
    results = multi_retrieve([vs], "query", k=6)
    assert results[0].page_content == "high score"


# ── format_docs ───────────────────────────────────────────────────────────────

def test_format_docs_joins_with_double_newline():
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert format_docs(docs) == "A\n\nB"


def test_format_docs_single_doc():
    docs = [Document(page_content="only")]
    assert format_docs(docs) == "only"


# ── Integration: real Bedrock call ────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID"),
    reason="No AWS credentials — skipping live Bedrock test",
)
def test_llm_bedrock_call_succeeds():
    """Smoke test: confirms the configured model ID is valid and reachable."""
    llm = _get_llm()
    response = llm.invoke("Reply with the single word: ok")
    assert response.content.strip()
