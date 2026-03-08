from fastapi import APIRouter, Depends, HTTPException, Header, status
from auth.keys import create_api_key
from config import settings

router = APIRouter(prefix="/admin", tags=["admin"], include_in_schema=False)


def require_master_key(x_master_key: str = Header(...)):
    if x_master_key != settings.api_master_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid master key")


@router.post("/keys")
def create_key(name: str, _: None = Depends(require_master_key)):
    """Create a new API key. The raw key is shown once — store it securely."""
    raw_key = create_api_key(name)
    return {
        "key": raw_key,
        "name": name,
        "note": "Store this key — it will not be shown again.",
    }
