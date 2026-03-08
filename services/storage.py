import hashlib
import sqlite3
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import settings

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_SUPPORTED = True
except ImportError:
    DOCX_SUPPORTED = False

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_document(path: str, suffix: str):
    """Load a document into LangChain Document objects. Returns None for unsupported types."""
    if suffix == ".pdf":
        return PyPDFLoader(path).load()
    elif suffix == ".txt":
        return TextLoader(path).load()
    elif suffix == ".docx" and DOCX_SUPPORTED:
        return Docx2txtLoader(path).load()
    return None


def save_uploaded_file(data: bytes, filename: str) -> Path:
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.documents_dir) / filename
    path.write_bytes(data)
    return path


def get_document(doc_id: str) -> dict | None:
    con = sqlite3.connect(settings.db_path)
    row = con.execute(
        "SELECT id, filename, chunks FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    con.close()
    return {"id": row[0], "filename": row[1], "chunks": row[2]} if row else None


def register_document(doc_id: str, filename: str, chunks: int):
    con = sqlite3.connect(settings.db_path)
    con.execute(
        "INSERT OR IGNORE INTO documents (id, filename, chunks) VALUES (?, ?, ?)",
        (doc_id, filename, chunks),
    )
    con.commit()
    con.close()
