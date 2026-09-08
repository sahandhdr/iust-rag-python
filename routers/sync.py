# routers/sync.py
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth_rbac import UserContext, get_qdrant_sync_manager
from create_database import wipe_and_recreate_collection
from dependencies import get_current_user
from utils.api_responser import ApiResponser

router = APIRouter(prefix="/sync", tags=["Sync - Laravel Integration"])

qdrant_sync = get_qdrant_sync_manager()


def _can_manage_sync(user: UserContext) -> bool:
    roles = set(user.roles or [])
    return bool(roles & {"admin", "developer", "superadmin"})


class WipeCollectionBody(BaseModel):
    confirm: bool = Field(
        False,
        description="Must be true to wipe the entire Qdrant collection",
    )


@router.get("/documents/{doc_uuid}")
async def get_document_chunks(
    doc_uuid: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: UserContext = Depends(get_current_user),
):
    """دریافت chunkهای یک سند (برای Laravel)"""
    if not _can_manage_sync(current_user):
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        results = qdrant_sync.search_by_doc_uuid(doc_uuid, limit=limit)
        if len(results) == 0:
            return ApiResponser.success_response(
                message="Retrieval Process Completed",
                data={
                    "doc_uuid": doc_uuid,
                    "total_chunks": len(results),
                    "chunks": results,
                    "result": "doc-notExists",
                },
            )

        return ApiResponser.success_response(
            message="Retrieval Process Completed",
            data={
                "doc_uuid": doc_uuid,
                "total_chunks": len(results),
                "chunks": results,
                "result": "doc-exists",
            },
        )
    except Exception as e:
        return ApiResponser.error_response(
            message="Retrieval Process Error",
            errors=str(e),
            status_code=500,
        )


@router.get("/documents/department/{department}")
async def list_documents_by_department(
    department: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: UserContext = Depends(get_current_user),
):
    """لیست اسناد یک دپارتمان"""
    if not _can_manage_sync(current_user):
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        results = qdrant_sync.list_documents_by_department(department, limit=limit)
        return ApiResponser.success_response(
            message=f"اسناد دپارتمان {department}",
            data={
                "department": department,
                "total": len(results),
                "results": results,
            },
        )
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در لیست اسناد",
            errors=str(e),
            status_code=500,
        )


@router.delete("/documents/{doc_uuid}")
async def sync_delete_document(
    doc_uuid: str,
    current_user: UserContext = Depends(get_current_user),
):
    """حذف سند - مخصوص فراخوانی Laravel"""
    if not _can_manage_sync(current_user):
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        success = qdrant_sync.delete_document_by_uuid(doc_uuid)
        if success:
            return ApiResponser.success_response(
                message=f"سند {doc_uuid} با موفقیت از Vector DB حذف شد",
                data={"doc_uuid": doc_uuid, "status": "deleted"},
            )
        return ApiResponser.error_response("حذف انجام نشد", status_code=500)
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در حذف سند",
            errors=str(e),
            status_code=500,
        )


@router.patch("/documents/{doc_uuid}/metadata")
async def sync_update_metadata(
    doc_uuid: str,
    payload: Dict[str, Any],
    current_user: UserContext = Depends(get_current_user),
):
    """بروزرسانی metadata - مخصوص Laravel"""
    if not _can_manage_sync(current_user):
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        success = qdrant_sync.update_document_metadata(doc_uuid, payload)
        if success:
            return ApiResponser.success_response(
                message=f"Metadata سند {doc_uuid} بروزرسانی شد"
            )
        return ApiResponser.error_response("بروزرسانی انجام نشد", status_code=500)
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در بروزرسانی metadata",
            errors=str(e),
            status_code=500,
        )


@router.post("/collection/wipe")
async def wipe_collection(
    body: WipeCollectionBody,
    current_user: UserContext = Depends(get_current_user),
):
    """
    Wipe entire Qdrant collection and recreate Hybrid schema.
    Requires confirm=true. Does not touch MySQL or Laravel storage.
    """
    if not _can_manage_sync(current_user):
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    if not body.confirm:
        return ApiResponser.error_response(
            message="برای پاک‌سازی کامل Collection باید confirm=true ارسال شود.",
            errors={"confirm": "required-true"},
            status_code=422,
        )

    try:
        data = wipe_and_recreate_collection()
        return ApiResponser.success_response(
            message="Collection wiped and hybrid schema recreated.",
            data=data,
        )
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در پاک‌سازی Collection",
            errors=str(e),
            status_code=500,
        )