from pydantic import BaseModel
from typing import Optional


class Source(BaseModel):
    document_id: str
    filename: str
    page: Optional[int] = None
    excerpt: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    conversation_id: str


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    document_ids: list[str]
    created_at: str
