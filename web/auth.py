"""API key authentication for the eval-harness web API.

Validates the X-API-Key header against the EVAL_HARNESS_API_KEY
environment variable. The health endpoint is excluded from auth.
"""

from __future__ import annotations

import os

from fastapi import Request, HTTPException


async def require_api_key(request: Request) -> None:
    """Dependency that enforces API key authentication.

    Compares the X-API-Key header against the EVAL_HARNESS_API_KEY env var.

    Args:
        request: The incoming FastAPI request.

    Raises:
        HTTPException: 401 if the key is missing or does not match.
    """
    expected = os.environ.get("EVAL_HARNESS_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="EVAL_HARNESS_API_KEY not configured")

    provided = request.headers.get("X-API-Key")
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
