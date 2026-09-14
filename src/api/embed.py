# embed.py
"""Lightweight query embedding for Laravel Semantic Cache."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.rbac import UserContext
from models.embeddings import get_embedding_function
from utils.api_responser import ApiResponser
from utils.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embed", tags=["Embed"])


class EmbedQueryBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@router.post("/query")
async def embed_query(
    body: EmbedQueryBody,
    current_user: UserContext = Depends(get_current_user),
):
    """
    Auth: Bearer or X-Internal-Key (same as other routes).
    Any authenticated caller may embed — used by Laravel gateway only in practice.
    """
    text = (body.text or "").strip()
    if not text:
        return ApiResponser.error_response(
            message="text is required",
            errors={"text": "required"},
            status_code=422,
        )

    try:
        emb = get_embedding_function()
        vector: List[float] = emb.embed_query(text)
        if not vector:
            return ApiResponser.error_response(
                message="empty-embedding",
                status_code=500,
            )
        return ApiResponser.success_response(
            message="embed-ok",
            data={
                "vector": vector,
                "dim": len(vector),
                "model": getattr(emb, "model", None),
            },
        )
    except Exception as exc:
        logger.exception("embed_query failed user=%s", getattr(current_user, "user_id", None))
        return ApiResponser.error_response(
            message="embed-failed",
            errors="internal-error",
            status_code=500,
        )