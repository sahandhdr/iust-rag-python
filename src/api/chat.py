# routers/chat.py
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from core.rag_engine import RAGEngine, rag_engine_instance
from auth.rbac import LaravelAuthenticator, UserContext
from ingest.documents import document_ingestor
from utils.api_responser import ApiResponser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])
optional_bearer = HTTPBearer(auto_error=False)

MAX_QUERY_LEN = 2000
MAX_SELECTED_TEXT_LEN = 50_000


def get_rag_engine() -> RAGEngine:
    return rag_engine_instance


class UserContextPayload(BaseModel):
    user_id: int = Field(..., ge=1)
    username: str = Field(..., min_length=1, max_length=255)
    roles: List[str] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    session_id: str = Field(..., min_length=1, max_length=64)
    selected_text: Optional[str] = Field(None, max_length=MAX_SELECTED_TEXT_LEN)
    msg_id: Optional[str] = Field(None, max_length=64)
    user_context: Optional[UserContextPayload] = None

    @field_validator("query", "session_id")
    @classmethod
    def strip_nonempty(cls, v: str) -> str:
        if v is None:
            raise ValueError("required")
        s = str(v).strip()
        if not s:
            raise ValueError("must not be blank")
        return s


async def resolve_user(
    request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> UserContext:
    if request.user_context is not None:
        uc = request.user_context
        return UserContext(
            user_id=uc.user_id,
            username=uc.username,
            roles=set(uc.roles or []),
            departments=set(uc.departments or []),
            permissions=set(uc.permissions or []),
        )
    if credentials and credentials.credentials:
        return await LaravelAuthenticator.verify_token(credentials.credentials)
    raise HTTPException(status_code=401, detail="توکن احراز هویت ارسال نشده است.")


def _sse_pack(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
    except Exception:
        logger.exception("Error processing chat request")
        return ApiResponser.error_response(
            message="خطایی در پردازش درخواست متنی رخ داد.",
            errors="internal-error",
            status_code=500,
        )


@router.post("/ask/stream")
async def ask_stream(
    request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
    engine: RAGEngine = Depends(get_rag_engine),
):
    start_time = time.time()
    current_user = await resolve_user(request, credentials)

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
            yield _sse_pack("error", {"message": "stream-failed"})

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
    query: str = Form(..., min_length=1, max_length=MAX_QUERY_LEN),
    session_id: str = Form(..., min_length=1, max_length=64),
    user_context: Optional[str] = Form(None),
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
    engine: RAGEngine = Depends(get_rag_engine),
):
    start_time = time.time()

    query = (query or "").strip()
    session_id = (session_id or "").strip()
    if not query:
        return ApiResponser.error_response(
            message="درخواست نامعتبر است.",
            errors={"query": "required"},
            status_code=422,
        )
    if not session_id:
        return ApiResponser.error_response(
            message="درخواست نامعتبر است.",
            errors={"session_id": "required"},
            status_code=422,
        )

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

    safe_name = os.path.basename(file.filename or "upload.bin")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
    temp_path = f"temp_chat_{safe_name}"

    try:
        content = await file.read()
        if not content:
            return ApiResponser.error_response(
                message="فایل خالی است.",
                status_code=422,
            )

        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        file_extension = os.path.splitext(safe_name)[1].lower()
        if file_extension in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}:
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing file chat request")
        return ApiResponser.error_response(
            message="خطایی در پردازش فایل رخ داد.",
            errors="internal-error",
            status_code=500,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass