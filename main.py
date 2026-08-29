# main.py
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, File, Form, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from auth_rbac import UserContext, qdrant_sync
from config import get_settings
from dependencies import get_current_user
from ingest_documents import document_ingestor
from routers.chat import router as chat_router
from routers.sync import router as sync_router
from utils.api_responser import ApiResponser

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="IUST RAG API",
    version="1.0.0",
    description="سیستم RAG مرکز کامپیوتر دانشگاه علم و صنعت - Phase 1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(sync_router, prefix=settings.api_prefix)


def _parse_json_list(raw: Optional[str], field_name: str) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid JSON array") from exc
    if not isinstance(data, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return [str(item).strip() for item in data if str(item).strip()]


def _can_manage_documents(user: UserContext) -> bool:
    roles = set(user.roles or [])
    return bool(roles & {"admin", "developer", "superadmin"})


def _safe_temp_name(filename: Optional[str]) -> str:
    base = os.path.basename(filename or "upload.bin")
    base = re.sub(r"[^a-zA-Z0-9._\-]", "_", base)
    return f"temp_ingest_{base}"


def _qdrant_chunk_count(doc_uuid: str) -> int:
    """
    search_by_doc_uuid returns a list of Qdrant points (scroll API).
    Older callers may also pass a dict shape from the sync router — support both.
    """
    info = qdrant_sync.search_by_doc_uuid(doc_uuid, limit=10_000)
    if isinstance(info, list):
        return len(info)
    if isinstance(info, dict):
        if info.get("result") == "doc-notExists":
            return 0
        try:
            return int(info.get("total_chunks") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


@app.get("/check")
async def check_health():
    try:
        return ApiResponser.success_response(
            message="Host is up and running",
            data={"status": "healthy", "version": "1.0.0"},
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return ApiResponser.error_response(
            message="خطا در بررسی سلامت سرویس",
            errors=str(exc),
            status_code=500,
        )


@app.post(f"{settings.api_prefix}/files/ingest")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    department: str = Form("public"),
    doc_uuid: Optional[str] = Form(None),
    roles: Optional[str] = Form(None),
    departments: Optional[str] = Form(None),
    permissions: Optional[str] = Form(None),
    status: str = Form("published"),
    version: int = Form(1),
    current_user: UserContext = Depends(get_current_user),
):
    if not _can_manage_documents(current_user):
        return ApiResponser.error_response(
            message="شما مجوز آپلود سند مرجع را ندارید.",
            status_code=403,
        )

    try:
        roles_list = _parse_json_list(roles, "roles")
        depts_list = _parse_json_list(departments, "departments")
        perms_list = _parse_json_list(permissions, "permissions")
    except ValueError as exc:
        return ApiResponser.error_response(
            message="پارامتر roles/departments/permissions نامعتبر است.",
            errors=str(exc),
            status_code=422,
        )

    status_norm = (status or "published").strip().lower()
    if status_norm not in {"draft", "published", "archived"}:
        return ApiResponser.error_response(
            message="status باید یکی از draft|published|archived باشد.",
            status_code=422,
        )

    temp_file_path = _safe_temp_name(file.filename)

    try:
        content = await file.read()
        if not content:
            return ApiResponser.error_response(message="فایل خالی است.", status_code=422)

        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        result = document_ingestor.process_single_file(
            file_path=temp_file_path,
            department=department or "public",
            doc_uuid=doc_uuid,
            roles=roles_list,
            departments=depts_list,
            permissions=perms_list,
            status=status_norm,
            version=version if isinstance(version, int) and version >= 1 else 1,
            overwrite=True,
        )

        if not result.get("success"):
            return ApiResponser.error_response(
                message="پردازش فایل با شکست مواجه شد. لطفاً محتوای فایل را بررسی کنید.",
                errors=result.get("error"),
                status_code=422,
            )

        return ApiResponser.success_response(
            message="فایل با موفقیت پردازش و به پایگاه دانش اضافه شد.",
            data={
                "filename": file.filename,
                "department": department or "public",
                "doc_uuid": result.get("doc_uuid") or doc_uuid or "auto-generated",
                "roles": result.get("roles") or roles_list,
                "departments": result.get("departments") or depts_list,
                "permissions": result.get("permissions") or perms_list,
                "status": result.get("status") or status_norm,
                "version": result.get("version") or version,
                "chunks": result.get("chunks"),
                "status_ingest": "ingested",
            },
        )
    except Exception as exc:
        logger.exception("Ingestion error for file %s (doc_uuid=%s)", file.filename, doc_uuid)
        return ApiResponser.error_response(
            message="خطا در پردازش فایل",
            errors=str(exc),
            status_code=500,
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as cleanup_err:
                logger.warning("Failed to cleanup temp file: %s", cleanup_err)


@app.delete(f"{settings.api_prefix}/files/{{doc_uuid}}")
async def delete_document_endpoint(
    doc_uuid: str = Path(..., description="شناسه یکتای سند (doc_uuid)"),
    current_user: UserContext = Depends(get_current_user),
):
    """حذف چانک‌های Qdrant. فایل Laravel و MD روی دیسک را پاک نمی‌کند."""
    if not _can_manage_documents(current_user):
        return ApiResponser.error_response("شما مجوز حذف سند را ندارید.", 403)

    try:
        existing = _qdrant_chunk_count(doc_uuid)
        if existing <= 0:
            return ApiResponser.error_response(
                message=f"سند {doc_uuid} در پایگاه دانش موجود نیست.",
                status_code=404,
                errors={"doc_uuid": doc_uuid, "result": "doc-notExists"},
            )

        success = qdrant_sync.delete_document_by_uuid(doc_uuid)
        if not success:
            return ApiResponser.error_response("حذف سند ناموفق بود.", 500)

        leftover = _qdrant_chunk_count(doc_uuid)
        if leftover > 0:
            logger.error("Delete reported success but chunks remain | doc_uuid=%s leftover=%s", doc_uuid, leftover)
            return ApiResponser.error_response("حذف سند ناقص بود.", 500)

        logger.info("Document deleted from Qdrant | doc_uuid=%s | by=%s | removed_chunks=%s",
                    doc_uuid, current_user.username, existing)
        return ApiResponser.success_response(
            message=f"سند {doc_uuid} با موفقیت حذف شد.",
            data={"doc_uuid": doc_uuid, "status": "deleted", "removed_chunks": existing},
        )
    except Exception:
        logger.exception("Delete error for doc_uuid=%s", doc_uuid)
        return ApiResponser.error_response("خطا در حذف سند.", 500)