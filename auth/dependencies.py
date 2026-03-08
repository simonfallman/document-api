from fastapi import Header, HTTPException, status
from auth.keys import hash_key, get_key_record, touch_last_used


async def validate_api_key(x_api_key: str = Header(...)) -> str:
    """FastAPI dependency. Validates X-API-Key header, returns key_hash."""
    key_hash = hash_key(x_api_key)
    record = get_key_record(key_hash)
    if not record or not record["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    touch_last_used(key_hash)
    return key_hash
