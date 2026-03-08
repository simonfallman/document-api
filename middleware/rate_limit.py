import time
import sqlite3
from fastapi import Depends, HTTPException, status
from auth.dependencies import validate_api_key
from config import settings


async def check_rate_limit(api_key_hash: str = Depends(validate_api_key)) -> str:
    """
    FastAPI dependency. Enforces a sliding-window rate limit per API key.
    Runs after validate_api_key, so api_key_hash is already verified.
    """
    now = time.time()
    window_start = now - settings.rate_limit_window_seconds

    con = sqlite3.connect(settings.db_path)

    # Prune expired entries for this key to keep the table small
    con.execute(
        "DELETE FROM rate_limit_requests WHERE api_key_hash = ? AND timestamp < ?",
        (api_key_hash, window_start),
    )

    count = con.execute(
        "SELECT COUNT(*) FROM rate_limit_requests WHERE api_key_hash = ? AND timestamp >= ?",
        (api_key_hash, window_start),
    ).fetchone()[0]

    if count >= settings.rate_limit_requests:
        con.close()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.rate_limit_requests} requests "
                f"per {settings.rate_limit_window_seconds}s"
            ),
        )

    con.execute(
        "INSERT INTO rate_limit_requests (api_key_hash, timestamp) VALUES (?, ?)",
        (api_key_hash, now),
    )
    con.commit()
    con.close()
    return api_key_hash
