"""Optional API-key guard for staff/operator endpoints."""

from __future__ import annotations

from fastapi import Header, HTTPException

from backend.retail_config import API_KEY


def require_staff_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
