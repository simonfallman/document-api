import json
import sqlite3
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from middleware.rate_limit import check_rate_limit
from models.requests import AskRequest
from models.responses import AskResponse, Source
from services.embeddings import get_vectorstores
from services.rag import build_chain
from services.storage import get_document
from config import settings

router = APIRouter(tags=["ask"])


def _log_usage(
    api_key_hash: str,
    document_ids: list[str],
    question: str,
    answer_length: int,
    duration_ms: int,
):
    con = sqlite3.connect(settings.db_path)
    con.execute(
        "INSERT INTO usage_logs "
        "(api_key_hash, endpoint, document_ids, question, answer_length, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (api_key_hash, "/ask", json.dumps(document_ids), question, answer_length, duration_ms),
    )
    con.commit()
    con.close()


def _ensure_conversation(conversation_id: str, api_key_hash: str, document_ids: list[str], title: str):
    con = sqlite3.connect(settings.db_path)
    con.execute(
        "INSERT OR IGNORE INTO conversations (id, api_key_hash, title, document_ids) VALUES (?, ?, ?, ?)",
        (conversation_id, api_key_hash, title, json.dumps(document_ids)),
    )
    con.commit()
    con.close()


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    api_key_hash: str = Depends(check_rate_limit),
):
    """
    Ask a question across one or more uploaded documents.
    Pass conversation_id to continue an existing conversation; omit to start a new one.
    """
    if not body.document_ids:
        raise HTTPException(status_code=400, detail="document_ids cannot be empty.")

    # Validate all document IDs exist before doing any work
    doc_records = {}
    for doc_id in body.document_ids:
        record = get_document(doc_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        doc_records[doc_id] = record

    conversation_id = body.conversation_id or str(uuid.uuid4())
    title = body.question[:60] + ("..." if len(body.question) > 60 else "")
    _ensure_conversation(conversation_id, api_key_hash, body.document_ids, title)

    vectorstores = get_vectorstores(body.document_ids)
    chain = build_chain(vectorstores)

    start = time.time()
    result = chain.invoke(
        {"input": body.question},
        config={"configurable": {"session_id": conversation_id}},
    )
    duration_ms = int((time.time() - start) * 1000)

    answer = result["answer"]

    sources = []
    for doc in result.get("context", []):
        meta = doc.metadata
        doc_id = meta.get("collection_hash", "")
        record = doc_records.get(doc_id) or get_document(doc_id)
        page = meta.get("page")
        sources.append(Source(
            document_id=doc_id,
            filename=record["filename"] if record else "",
            page=int(page) + 1 if page is not None else None,
            excerpt=doc.page_content[:200],
        ))

    _log_usage(api_key_hash, body.document_ids, body.question, len(answer), duration_ms)

    return AskResponse(answer=answer, sources=sources, conversation_id=conversation_id)
