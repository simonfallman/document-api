# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document Intelligence REST API — a FastAPI backend that exposes RAG-based document Q&A as a REST API. Developers send HTTP requests with documents and questions, get back JSON answers with source citations. Stack: Python 3.11, FastAPI, LangChain, AWS Bedrock, ChromaDB, SQLite, Docker, GitHub Actions.

## Related Project

The AI Document Chatbot (Streamlit UI) lives at `../ai-document-chatbot/`. The core RAG logic (embeddings, chunking, retrieval) is intentionally similar — this project exposes the same intelligence as a proper API instead of a UI.

## Development Commands

```bash
# Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest httpx

# Run locally
uvicorn main:app --reload

# Run all tests (no AWS credentials needed — Bedrock is mocked)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_ask.py::test_ask_returns_answer -v

# Docker (local dev)
docker compose up --build

# Create first API key (replace master key value)
curl -X POST "http://localhost:8000/admin/keys?name=mykey" \
  -H "X-Master-Key: your-api-master-key"
```

Swagger docs at `http://localhost:8000/docs` when running locally.

## Environment

Copy `.env.example` to `.env` and fill in:
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` — required for Bedrock
- `API_MASTER_KEY` — used to create user API keys via `POST /admin/keys`

## Architecture

### Module layout

```
main.py            FastAPI app, lifespan (calls init_db on startup)
config.py          pydantic-settings Settings singleton — all config lives here
database.py        SQLite schema creation (init_db)

models/            Pydantic request/response shapes
auth/
  keys.py          Key generation (sha256 hash stored, raw shown once), SQLite CRUD
  dependencies.py  validate_api_key FastAPI dependency → returns key_hash
middleware/
  rate_limit.py    check_rate_limit dependency — sliding window over SQLite
services/
  storage.py       file_hash (MD5), load_document, save_uploaded_file, get/register document
  embeddings.py    BedrockEmbeddings, build_vectorstore, get_vectorstore(s)
  rag.py           build_chain, multi_retrieve, tool_summarize, tool_faq
routers/
  documents.py     POST /documents/upload
  ask.py           POST /ask (also logs to usage_logs)
  conversations.py GET /conversations, DELETE /conversations/{id}
  admin.py         POST /admin/keys (hidden from Swagger, protected by X-Master-Key)
tests/
  conftest.py      isolated SQLite per test, auth dependency overrides, Bedrock mocks
```

### Key design decisions

**Settings patching in tests** — `conftest.py` uses `monkeypatch.setattr(settings, "db_path", ...)` to redirect the `settings` singleton to temp paths. This works because all service functions access `settings.db_path` at call time, not import time.

**Auth dependency chain** — `check_rate_limit` depends on `validate_api_key`, so rate-limited routes only need `Depends(check_rate_limit)`. In tests both are overridden together via `app.dependency_overrides`.

**Conversation ownership** — conversations store `api_key_hash`, so `GET /conversations` only returns the calling key's conversations.

**Document idempotency** — `document_id` is MD5 of file bytes. Uploading the same file twice returns the cached result without re-embedding.

### RAG Pipeline

Same as `../ai-document-chatbot/app.py` with one key change:

| Chatbot | API |
|---|---|
| `lambda session_id: st.session_state.chat_history` | `lambda session_id: SQLChatMessageHistory(session_id, f"sqlite:///{settings.db_path}")` |

The chain returns `{"answer": str, "context": list[Document]}`. Chunk metadata includes `collection_hash` (= document_id), `document_name`, `page`, `chunk_index`.

### SQLite tables

- `documents` — id (MD5), filename, chunks
- `conversations` — id (UUID), api_key_hash, title, document_ids (JSON), created_at
- `message_store` — LangChain SQLChatMessageHistory writes here (session_id, message)
- `api_keys` — key_hash (SHA-256), name, is_active, last_used_at
- `rate_limit_requests` — api_key_hash, timestamp (unix float), pruned on each request
- `usage_logs` — api_key_hash, endpoint, document_ids, question, answer_length, duration_ms

All data files live under `./data/` (mounted as a Docker volume in production).

## Endpoints

- `POST /documents/upload` — multipart file upload → `{document_id, filename, chunks}`
- `POST /ask` — `{document_ids, question, conversation_id?}` → `{answer, sources, conversation_id}`
- `GET /conversations` — list conversations for authenticated key
- `DELETE /conversations/{id}` — delete conversation + message history
- `POST /admin/keys?name=x` — create API key (needs `X-Master-Key` header, not in Swagger)
- `GET /health` — health check

## Deployment

- Hetzner VPS at 89.167.107.181; Nginx routes `simonfallman.xyz/api` → port 8000
- Named Docker container `document-api` (avoids conflicting with the chatbot container)
- Volume mount: `/root/document-api/data:/app/data`
- GitHub Actions: `test.yml` runs pytest on every push; `deploy.yml` runs tests then SSH-deploys on push to `main`
- Required GitHub secret: `DOCUMENT_API_SSH_KEY`

## Commit Rules

- Never add Co-Authored-By or any Claude/AI attribution to commit messages
