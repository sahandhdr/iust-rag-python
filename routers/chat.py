# routers/chat.py
"""
Chat routes:
  POST /chat/ask            — one-shot JSON
  POST /chat/ask_with_file  — one-shot JSON + uploaded file
  POST /chat/ask/stream     — SSE (meta, sources, token*, done|error)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from RAGEngine import RAGEngine, rag_engine_instance
from auth_rbac import LaravelAuthenticator, UserContext
from ingest_documents import document_ingestor
from utils.api_responser import ApiResponser
from vision_helper import vision_helper_instance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])
optional_bearer = HTTPBearer(auto_error=False)


def get_rag_engine() -> RAGEngine:
    return rag_engine_instance


class UserContextPayload(BaseModel):
    user_id: int
    username: str
    roles: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str
    session_id: str
    selected_text: Optional[str] = None
    msg_id: Optional[str] = None  # accepted but not used for Qdrant filter
    user_context: Optional[UserContextPayload] = None


async def resolve_user(
    request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> UserContext:
    """
    1) Laravel user_context → no callback
    2) else Bearer → verify_token
    """
    if request.user_context is not None:
        try:
            return UserContext.model_validate(request.user_context.model_dump())
        except Exception as exc:
            logger.exception("Invalid user_context from Laravel proxy")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="اطلاعات هویتی نامعتبر است.",
            ) from exc

    if credentials is not None and credentials.credentials:
        return await LaravelAuthenticator.verify_token(credentials.credentials)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="توکن احراز هویت ارسال نشده است.",
    )


def _sse_pack(event: str, data: object) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/ask")
async def ask_question(
    request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
    engine: RAGEngine = Depends(get_rag_engine),
):
    start_time = time.time()
    current_user = await resolve_user(request, credentials)

    logger.info(
        "Received text query from user %s for session %s (via_context=%s)",
        current_user.username,
        request.session_id,
        request.user_context is not None,
    )

    try:
        answer, sources = await engine.query(
            question=request.query,
            session_id=request.session_id,
            user_context=current_user,
            user_file_content=request.selected_text,
            msg_id=None,
        )

        processing_time = round(time.time() - start_time, 2)
        return ApiResponser.success_response(
            message="پاسخ با موفقیت تولید شد.",
            data={
                "answer": answer,
                "sources": sources,
                "session_id": request.session_id,
                "processing_time": processing_time,
                "user_department": sorted(current_user.departments),
            },
        )
    except Exception as e:
        logger.exception("Error processing chat request:")
        return ApiResponser.error_response(
            message="خطایی در پردازش درخواست متنی رخ داد.",
            errors=str(e),
            status_code=500,
        )


@router.post("/ask/stream")
async def ask_stream(
    request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
    engine: RAGEngine = Depends(get_rag_engine),
):
    """
    SSE stream. Intended to be proxied by Laravel (not called directly by browser in Phase 1).
    Events: meta → sources → token* → done | error
    """
    start_time = time.time()
    current_user = await resolve_user(request, credentials)

    if not request.session_id or not str(request.session_id).strip():
        raise HTTPException(status_code=422, detail="session_id الزامی است.")
    if not request.query or not str(request.query).strip():
        raise HTTPException(status_code=422, detail="query الزامی است.")

    logger.info(
        "Stream query from user %s session %s (via_context=%s)",
        current_user.username,
        request.session_id,
        request.user_context is not None,
    )

    async def event_generator():
        try:
            yield _sse_pack(
                "meta",
                {
                    "session_id": request.session_id,
                    "user_id": current_user.user_id,
                },
            )

            async for kind, payload in engine.query_stream(
                question=request.query,
                session_id=request.session_id,
                user_context=current_user,
                user_file_content=request.selected_text,
            ):
                if kind == "sources":
                    yield _sse_pack("sources", payload)
                elif kind == "token":
                    yield _sse_pack("token", {"t": payload})
                elif kind == "done":
                    processing_time = round(time.time() - start_time, 2)
                    yield _sse_pack(
                        "done",
                        {
                            "answer": payload.get("answer", ""),
                            "processing_time": processing_time,
                            "session_id": request.session_id,
                        },
                    )
        except Exception as exc:
            logger.exception("ask/stream failed")
            yield _sse_pack("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ask_with_file")
async def ask_with_file(
    query: str = Form(...),
    session_id: str = Form(...),
    user_context: Optional[str] = Form(None),
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
    engine: RAGEngine = Depends(get_rag_engine),
):
    start_time = time.time()

    if user_context:
        try:
            payload = json.loads(user_context)
            current_user = UserContext.model_validate(payload)
        except Exception as exc:
            logger.exception("Invalid user_context form field")
            raise HTTPException(
                status_code=401, detail="اطلاعات هویتی نامعتبر است."
            ) from exc
    elif credentials and credentials.credentials:
        current_user = await LaravelAuthenticator.verify_token(credentials.credentials)
    else:
        raise HTTPException(
            status_code=401, detail="توکن احراز هویت ارسال نشده است."
        )

    logger.info(
        "Received file query from user %s. File: %s",
        current_user.username,
        file.filename,
    )

    temp_path = f"temp_chat_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        file_extension = os.path.splitext(file.filename or "")[1].lower()

        if file_extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
            extracted_text = vision_helper_instance.analyze_image(
                temp_path, analysis_mode="ocr"
            )
        else:
            extracted_text = document_ingestor.extract_text_from_file(temp_path)

        if not extracted_text or not str(extracted_text).strip():
            return ApiResponser.error_response(
                message="متن قابل استخراج از فایل یافت نشد.",
                status_code=422,
            )

        answer, sources = await engine.query(
            question=query,
            session_id=session_id,
            user_context=current_user,
            user_file_content=extracted_text,
            msg_id=None,
        )

        processing_time = round(time.time() - start_time, 2)
        return ApiResponser.success_response(
            message="پاسخ با موفقیت تولید شد.",
            data={
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "processing_time": processing_time,
                "user_department": sorted(current_user.departments),
                "file_processed": file.filename,
            },
        )
    except Exception as e:
        logger.error("Error processing file chat request: %s", str(e), exc_info=True)
        return ApiResponser.error_response(
            message="خطایی در پردازش فایل رخ داد.",
            errors=str(e),
            status_code=500,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass