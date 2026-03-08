from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from middleware.rate_limit import check_rate_limit
from models.responses import UploadResponse
from services.embeddings import build_vectorstore
from services.storage import (
    file_hash,
    get_document,
    load_document,
    register_document,
    save_uploaded_file,
    SUPPORTED_SUFFIXES,
)
from config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key_hash: str = Depends(check_rate_limit),
):
    """
    Upload a document. Returns a document_id (MD5 hash of file contents).
    Uploading the same file twice returns the cached result immediately.
    Supported formats: PDF, TXT, DOCX.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size // 1024 // 1024}MB.",
        )

    doc_id = file_hash(data)

    # Idempotent: same file content always produces the same doc_id
    existing = get_document(doc_id)
    if existing:
        return UploadResponse(
            document_id=doc_id,
            filename=existing["filename"],
            chunks=existing["chunks"],
        )

    save_path = save_uploaded_file(data, file.filename)
    docs = load_document(str(save_path), suffix)
    if docs is None:
        raise HTTPException(status_code=422, detail="Could not parse document content.")

    chunks = build_vectorstore(docs, collection_name=doc_id, document_name=file.filename)
    register_document(doc_id, file.filename, chunks)

    return UploadResponse(document_id=doc_id, filename=file.filename, chunks=chunks)
