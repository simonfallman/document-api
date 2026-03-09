from fastapi import APIRouter, Depends
from pydantic import BaseModel

from middleware.rate_limit import check_rate_limit
from services.rag import _get_llm

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    context: str
    question: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    api_key_hash: str = Depends(check_rate_limit),
):
    """Ask a question about any text without uploading a document."""
    llm = _get_llm()
    prompt = (
        f"Answer the following question based on the text below. "
        f"Be concise and accurate.\n\n"
        f"Text:\n{body.context}\n\n"
        f"Question: {body.question}"
    )
    response = llm.invoke(prompt)
    return ChatResponse(answer=response.content)
