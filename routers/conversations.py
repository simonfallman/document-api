import json
import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import validate_api_key
from models.responses import ConversationResponse
from config import settings

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
def list_conversations(api_key_hash: str = Depends(validate_api_key)):
    """List all conversations belonging to the authenticated API key, newest first."""
    con = sqlite3.connect(settings.db_path)
    rows = con.execute(
        "SELECT id, title, document_ids, created_at FROM conversations "
        "WHERE api_key_hash = ? ORDER BY created_at DESC",
        (api_key_hash,),
    ).fetchall()
    con.close()
    return [
        ConversationResponse(
            id=r[0],
            title=r[1],
            document_ids=json.loads(r[2]) if r[2] else [],
            created_at=r[3],
        )
        for r in rows
    ]


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, api_key_hash: str = Depends(validate_api_key)):
    """Delete a conversation and its full message history."""
    con = sqlite3.connect(settings.db_path)

    result = con.execute(
        "DELETE FROM conversations WHERE id = ? AND api_key_hash = ?",
        (conversation_id, api_key_hash),
    )
    if result.rowcount == 0:
        con.close()
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Delete LangChain message history for this conversation
    try:
        con.execute("DELETE FROM message_store WHERE session_id = ?", (conversation_id,))
    except sqlite3.OperationalError:
        pass  # message_store may not exist if no messages were sent

    con.commit()
    con.close()
