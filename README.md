# Document Intelligence API

A production-ready REST API that lets developers query documents using natural language. Upload a PDF, Word doc, or text file and ask questions — get back answers with source citations and full conversation history.

Live at **[simonfallman.xyz/api/docs](https://simonfallman.xyz/api/docs)**

## What it does

- Upload documents (PDF, DOCX, TXT)
- Ask questions across one or multiple documents
- Maintains conversation history across requests
- Returns answers with cited source excerpts

## Stack

- **FastAPI** — REST API with auto-generated Swagger docs
- **AWS Bedrock** — Amazon Titan for embeddings, Claude 3.5 Haiku for generation
- **ChromaDB** — vector store, one collection per document (MD5-keyed, idempotent)
- **LangChain** — RAG pipeline with LCEL and conversation history
- **SQLite** — API keys, conversations, rate limiting, usage logs
- **Docker** — containerised, deployed on a Hetzner VPS
- **GitHub Actions** — CI/CD, tests run on every push, auto-deploys on main

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload a file, get back a `document_id` |
| `POST` | `/ask` | Ask a question across documents, get answer + sources |
| `GET` | `/conversations` | List your conversations |
| `DELETE` | `/conversations/{id}` | Delete a conversation |
| `GET` | `/health` | Health check |

All endpoints require an `X-API-Key` header.

## Try it

A demo key is available for testing — **[contact me](https://www.linkedin.com/in/simonfallman)** and I'll send it over.

Or explore the live Swagger docs at **[simonfallman.xyz/api/docs](https://simonfallman.xyz/api/docs)**.

```bash
# Upload a document
curl -X POST "https://simonfallman.xyz/api/documents/upload" \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@document.pdf"

# Ask a question
curl -X POST "https://simonfallman.xyz/api/ask" \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["<document_id>"], "question": "What is this document about?"}'
```

## Architecture

Requests flow through API key auth → rate limiting (60 req/min per key) → the RAG pipeline:

1. Uploaded files are chunked with `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap)
2. Chunks are embedded via Amazon Titan and stored in ChromaDB
3. On each question, relevant chunks are retrieved across all selected documents
4. Claude 3.5 Haiku generates an answer grounded in the retrieved context
5. Conversation history is persisted in SQLite via LangChain's `SQLChatMessageHistory`

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # fill in AWS credentials and API_MASTER_KEY

uvicorn main:app --reload
# Swagger docs at http://localhost:8000/docs
```

```bash
# Run tests (AWS credentials optional — skips live Bedrock test if not set)
python -m pytest tests/ -v
```

## Contact

Built by Simon Fallman — [linkedin.com/in/simonfallman](https://www.linkedin.com/in/simonfallman)
