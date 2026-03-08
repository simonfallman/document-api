from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    document_ids: list[str]
    question: str
    conversation_id: Optional[str] = None
