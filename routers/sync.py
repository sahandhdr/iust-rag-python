# routers/sync.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any

from dependencies import get_current_user
from auth_rbac import UserContext, get_qdrant_sync_manager
from utils.api_responser import ApiResponser

router = APIRouter(prefix="/sync", tags=["Sync - Laravel Integration"])

qdrant_sync = get_qdrant_sync_manager()


@router.get("/documents/{doc_uuid}")
async def get_document_chunks(
    doc_uuid: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: UserContext = Depends(get_current_user)
):
    """دریافت chunkهای یک سند (برای Laravel)"""
    if "superadmin" not in current_user.roles and "admin" not in current_user.roles:
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
                    "result" : "doc-notExists"
                }
            )

        elif len(results) >= 0:
            return ApiResponser.success_response(
                message="Retrieval Process Completed",
                data={
                    "doc_uuid": doc_uuid,
                    "total_chunks": len(results),
                    "chunks": results,
                    "result" : "doc-exists"
                }
            )

    except Exception as e:
        return ApiResponser.error_response(
            message="Retrieval Process Error",
            errors=str(e),
            status_code=500
        )


@router.get("/documents/department/{department}")
async def list_documents_by_department(
    department: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: UserContext = Depends(get_current_user)
):
    """لیست اسناد یک دپارتمان"""
    if "superadmin" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        results = qdrant_sync.list_documents_by_department(department, limit=limit)
        return ApiResponser.success_response(
            message=f"اسناد دپارتمان {department}",
            data={
                "department": department,
                "total": len(results),
                "results": results
            }
        )
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در لیست اسناد",
            errors=str(e),
            status_code=500
        )


@router.delete("/documents/{doc_uuid}")
async def sync_delete_document(
    doc_uuid: str,
    current_user: UserContext = Depends(get_current_user)
):
    """حذف سند - مخصوص فراخوانی Laravel"""
    if "superadmin" not in current_user.roles and "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="دسترسی ندارید")

    try:
        success = qdrant_sync.delete_document_by_uuid(doc_uuid)
        if success:
            return ApiResponser.success_response(
                message=f"سند {doc_uuid} با موفقیت از Vector DB حذف شد",
                data={"doc_uuid": doc_uuid, "status": "deleted"}
            )
        return ApiResponser.error_response("حذف انجام نشد", status_code=500)
    except Exception as e:
        return ApiResponser.error_response(
            message="خطا در حذف سند",
            errors=str(e),
            status_code=500
        )


@router.patch("/documents/{doc_uuid}/metadata")
async def sync_update_metadata(
    doc_uuid: str,
    payload: Dict[str, Any],
    current_user: UserContext = Depends(get_current_user)
):
    """بروزرسانی metadata - مخصوص Laravel"""
    if "superadmin" not in current_user.roles and "admin" not in current_user.roles:
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
            status_code=500
        )