"""
Auth + RBAC façade for IUST RAG AI Worker.

Responsibilities:
  - UserContext: normalized identity payload from Laravel (or mock tokens)
  - LaravelAuthenticator: token verification (mock now; real HTTP later)
  - RBACManager: delegates document ACL to access_control.DocumentAccessPolicy
  - QdrantSyncManager: sync document metadata / delete by doc_uuid

Design rules:
  - Fail-closed on invalid identity
  - Laravel is the source of truth for identity
  - Document access decisions are config-driven (or / and / hybrid)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------

class UserContext(BaseModel):
    """
    Normalized user context for auth + document ACL.

    - roles / departments / permissions stored as sets
    - all string tags: strip + lowercase
    - Laravel-ready shape (map 1:1 from API payload)
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=False,
        validate_assignment=True,
    )

    user_id: int
    username: str
    roles: Set[str] = Field(default_factory=set)
    departments: Set[str] = Field(default_factory=set)
    permissions: Set[str] = Field(default_factory=set)

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        for key in ("roles", "departments", "permissions"):
            if key in data and isinstance(data[key], set):
                data[key] = sorted(data[key])
        return data

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("user_id must be an integer")
        if value <= 0:
            raise ValueError("user_id must be a positive integer")
        return value

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("username must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("username must not be empty")
        return normalized

    @field_validator("roles", "departments", "permissions", mode="before")
    @classmethod
    def normalize_string_collections(cls, value: Any) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Iterable) or isinstance(value, (bytes, dict)):
            raise ValueError("Expected an iterable of strings")
        normalized_items: Set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All collection items must be strings")
            normalized = item.strip().lower()
            if normalized:
                normalized_items.add(normalized)
        return normalized_items

    @model_validator(mode="after")
    def validate_domain_rules(self) -> "UserContext":
        if not self.roles:
            raise ValueError("UserContext.roles must not be empty")
        return self

    def has_role(self, role: str) -> bool:
        return role.strip().lower() in self.roles

    def has_any_role(self, roles: Iterable[str]) -> bool:
        wanted = {r.strip().lower() for r in roles if isinstance(r, str) and r.strip()}
        return bool(self.roles & wanted)

    def has_permission(self, permission: str) -> bool:
        normalized = permission.strip().lower()
        return "all" in self.permissions or normalized in self.permissions

    def has_any_permission(self, permissions: Iterable[str]) -> bool:
        for permission in permissions:
            if self.has_permission(permission):
                return True
        return False

    def belongs_to_department(self, department: str) -> bool:
        return normalize_tag(department) in self.departments


def normalize_tag(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("tag must be a string")
    return value.strip().lower()


# ---------------------------------------------------------------------------
# Laravel authenticator (mock → real HTTP in integration phase)
# ---------------------------------------------------------------------------

# class LaravelAuthenticator:
#     """
#     Token verification.
#
#     Current phase: in-memory mock tokens (fail-closed).
#     Next phase: HTTP call to Laravel verify-token endpoint.
#     """
#
#     VALID_TOKENS: Dict[str, Dict[str, Any]] = {
#         "admin_token": {
#             "user_id": 1,
#             "username": "admin",
#             "roles": {"admin", "superadmin"},
#             "departments": {"it", "hr", "ce_dept"},
#             "permissions": {"all", "documents.read.all", "rbac.bypass"},
#         },
#         "staff_token": {
#             "user_id": 3,
#             "username": "staff1",
#             "roles": {"staff"},
#             "departments": {"it"},
#             "permissions": {"documents.read.public", "documents.read.department"},
#         },
#         "user_token": {
#             "user_id": 2,
#             "username": "student1",
#             "roles": {"public"},
#             "departments": {"ce_dept"},
#             "permissions": {"documents.read.public"},
#         },
#     }
#
#     @staticmethod
#     async def verify_token(token: str) -> UserContext:
#         if not token or not isinstance(token, str) or not token.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="توکن احراز هویت ارسال نشده است.",
#             )
#
#         normalized_token = token.strip()
#         user_payload = LaravelAuthenticator.VALID_TOKENS.get(normalized_token)
#         if not user_payload:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="توکن نامعتبر است.",
#             )
#
#         try:
#             return UserContext.model_validate(user_payload)
#         except Exception as exc:
#             logger.exception("Failed to validate user payload from authenticator")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="اطلاعات هویتی کاربر معتبر نیست.",
#             ) from exc

class LaravelAuthenticator:
    """
    Token verification via Laravel Sanctum.
    GET/POST {laravel.base_url}{verify_token_endpoint}
    Authorization: Bearer <token>
    """

    @staticmethod
    async def verify_token(token: str) -> UserContext:
        if not token or not isinstance(token, str) or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="توکن احراز هویت ارسال نشده است.",
            )

        normalized_token = token.strip()
        settings = get_settings()
        base_url = settings.laravel.base_url.rstrip("/")
        endpoint = settings.laravel.verify_token_endpoint
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = f"{base_url}{endpoint}"

        try:
            import httpx
        except ImportError as exc:
            logger.error("httpx is required for Laravel token verification")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="پیکربندی احراز هویت ناقص است.",
            ) from exc

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {normalized_token}",
                        "Accept": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )
        except httpx.RequestError as exc:
            logger.exception("Laravel verify-token connection error url=%s err=%s", url, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="سرویس احراز هویت در دسترس نیست.",
            ) from exc

        if response.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="توکن نامعتبر است.",
            )

        if response.status_code >= 400:
            logger.error(
                "Laravel verify-token failed status=%s body=%s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="توکن نامعتبر است.",
            )

        try:
            body = response.json()
        except Exception as exc:
            logger.exception("Invalid JSON from Laravel verify-token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="پاسخ احراز هویت نامعتبر است.",
            ) from exc

        # سازگار با successResponse لاراول: data در ریشه یا داخل data
        payload = body.get("data") if isinstance(body, dict) else None
        if not isinstance(payload, dict):
            payload = body if isinstance(body, dict) else {}

        # اگر کل پاسخ successResponse باشد و user داخل data باشد
        if "user_id" not in payload and isinstance(body.get("data"), dict):
            payload = body["data"]

        try:
            return UserContext.model_validate(
                {
                    "user_id": payload.get("user_id"),
                    "username": payload.get("username"),
                    "roles": payload.get("roles") or [],
                    "departments": payload.get("departments") or [],
                    "permissions": payload.get("permissions") or [],
                }
            )
        except Exception as exc:
            logger.exception("Failed to map Laravel identity to UserContext")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="اطلاعات هویتی کاربر معتبر نیست.",
            ) from exc

# ---------------------------------------------------------------------------
# RBACManager — façade over DocumentAccessPolicy
# ---------------------------------------------------------------------------

class RBACManager:
    """
    Document ACL façade.

    Primary engine: access_control.DocumentAccessPolicy (or / and / hybrid).
    Keeps a thin legacy fallback if Policy cannot be constructed.
    """

    def __init__(self) -> None:
        self._policy = None
        self._settings = get_settings()
        try:
            from access_control.policy import DocumentAccessPolicy

            self._policy = DocumentAccessPolicy.from_settings(self._settings)
        except Exception as exc:
            logger.warning(
                "DocumentAccessPolicy init failed; legacy department filter only. err=%s",
                exc,
            )

    @property
    def policy(self):
        return self._policy

    @staticmethod
    def normalize_department(department: Optional[str]) -> str:
        return normalize_tag(department)

    @staticmethod
    def normalize_permission(permission: Optional[str]) -> str:
        return normalize_tag(permission)

    def is_admin(self, user_context: UserContext) -> bool:
        if self._policy is not None:
            return self._policy.is_admin(user_context)
        admin_roles = {
            normalize_tag(r) for r in self._settings.rbac.admin_roles
        }
        return bool(user_context.roles & admin_roles)

    def can_bypass(self, user_context: UserContext) -> bool:
        if self._policy is not None:
            return self._policy.can_bypass(user_context)
        if not self.is_admin(user_context):
            return False
        bypass = {
            normalize_tag(p) for p in self._settings.rbac.admin_bypass_permissions
        }
        if "all" in user_context.permissions:
            return True
        return bool(user_context.permissions & bypass)

    # backward-compatible alias
    def can_bypass_department_filter(self, user_context: UserContext) -> bool:
        return self.can_bypass(user_context)

    def evaluate_document_access(
        self,
        user_context: UserContext,
        *,
        doc_roles: Optional[Iterable[str]] = None,
        doc_departments: Optional[Iterable[str]] = None,
        doc_permissions: Optional[Iterable[str]] = None,
        doc_status: Optional[str] = "published",
    ) -> bool:
        """Full multi-tag access check (preferred API)."""
        if self._policy is not None:
            return self._policy.evaluate_document_access(
                user_context,
                doc_roles=doc_roles,
                doc_departments=doc_departments,
                doc_permissions=doc_permissions,
                doc_status=doc_status,
            )
        # legacy: department-only
        depts = list(doc_departments or [])
        if not depts:
            return False
        return self.has_access(user_context, depts[0])

    def has_access(self, user_context: UserContext, doc_department: str) -> bool:
        """Legacy single-department check."""
        return self.evaluate_document_access(
            user_context,
            doc_roles=[],
            doc_departments=[doc_department] if doc_department else [],
            doc_status="published",
        )

    def build_qdrant_filter(self, user_context: UserContext) -> Optional[rest.Filter]:
        """
        Build Qdrant filter for retrieval.
        Prefer Policy; fall back to legacy department-only filter.
        """
        if self._policy is not None:
            return self._policy.build_qdrant_filter(user_context)

        # ----- legacy fallback (department scalar only) -----
        if self.can_bypass(user_context):
            if self._settings.rbac.require_published:
                return rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.status",
                            match=rest.MatchValue(
                                value=self._settings.rbac.published_status_value
                            ),
                        )
                    ]
                )
            return None

        allowed: Set[str] = set(user_context.departments)
        # public role users can always see public department tag in legacy mode
        if user_context.has_role("public") or user_context.has_permission(
            "documents.read.public"
        ):
            allowed.add("public")

        if not allowed:
            return rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.department",
                        match=rest.MatchValue(value="__forbidden__"),
                    )
                ]
            )

        must: List[Any] = [
            rest.FieldCondition(
                key="metadata.department",
                match=rest.MatchAny(any=sorted(allowed)),
            )
        ]
        if self._settings.rbac.require_published:
            must.append(
                rest.FieldCondition(
                    key="metadata.status",
                    match=rest.MatchValue(
                        value=self._settings.rbac.published_status_value
                    ),
                )
            )
        return rest.Filter(must=must)


# ---------------------------------------------------------------------------
# Qdrant sync
# ---------------------------------------------------------------------------

class QdrantSyncError(Exception):
    """Domain error for Qdrant sync operations."""


class QdrantSyncManager:
    """
    Sync document lifecycle between Laravel metadata and Qdrant payloads.

    - fail-closed (raise, never silent)
    - protected fields cannot be overwritten (doc_uuid)
    - supports multi-tag metadata: roles, departments, permissions, status, version
    """

    PROTECTED_METADATA_FIELDS = frozenset({"doc_uuid"})
    LIST_METADATA_FIELDS = frozenset({"roles", "departments", "permissions"})

    def __init__(self, client: Optional[QdrantClient] = None) -> None:
        self.settings = get_settings()
        qdrant_path = self.settings.db.qdrant_path
        qdrant_url = (self.settings.db.qdrant_url or "").strip()
        self.collection_name = self.settings.db.qdrant_collection

        try:
            if client is not None:
                self.client = client
            elif qdrant_url:
                self.client = QdrantClient(url=qdrant_url)
            else:
                os.makedirs(qdrant_path, exist_ok=True)
                self.client = QdrantClient(path=qdrant_path)
        except Exception as exc:
            logger.exception("Failed to initialize QdrantClient")
            raise QdrantSyncError("Qdrant client initialization failed.") from exc

    @staticmethod
    def validate_doc_uuid(doc_uuid: str) -> str:
        if not isinstance(doc_uuid, str):
            raise QdrantSyncError("doc_uuid must be a string")
        normalized = doc_uuid.strip()
        if not normalized:
            raise QdrantSyncError("doc_uuid must not be empty")
        return normalized

    def validate_metadata_update(self, new_metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(new_metadata, dict):
            raise QdrantSyncError("new_metadata must be a dictionary")

        normalized: Dict[str, Any] = {}
        for key, value in new_metadata.items():
            if not isinstance(key, str):
                raise QdrantSyncError("All metadata keys must be strings")
            normalized_key = key.strip()
            if not normalized_key:
                raise QdrantSyncError("Metadata keys must not be empty")
            if normalized_key in self.PROTECTED_METADATA_FIELDS:
                raise QdrantSyncError(
                    f"{normalized_key} cannot be overwritten via metadata update"
                )

            if normalized_key in self.LIST_METADATA_FIELDS:
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list):
                    raise QdrantSyncError(
                        f"metadata.{normalized_key} must be a list of strings"
                    )
                normalized[normalized_key] = sorted(
                    {
                        normalize_tag(item)
                        for item in value
                        if isinstance(item, str) and item.strip()
                    }
                )
            elif normalized_key == "department":
                # legacy scalar
                tag = normalize_tag(value if isinstance(value, str) else str(value))
                if not tag:
                    raise QdrantSyncError("metadata.department must not be empty")
                normalized[normalized_key] = tag
            elif normalized_key == "status":
                normalized[normalized_key] = normalize_tag(
                    value if isinstance(value, str) else str(value)
                )
            elif normalized_key == "version":
                try:
                    normalized[normalized_key] = int(value)
                except (TypeError, ValueError) as exc:
                    raise QdrantSyncError("metadata.version must be an integer") from exc
            else:
                normalized[normalized_key] = value

        return normalized

    def _doc_selector(self, doc_uuid: str) -> rest.FilterSelector:
        validated_uuid = self.validate_doc_uuid(doc_uuid)
        return rest.FilterSelector(
            filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.doc_uuid",
                        match=rest.MatchValue(value=validated_uuid),
                    )
                ]
            )
        )

    def delete_document_by_uuid(self, doc_uuid: str) -> bool:
        validated_uuid = self.validate_doc_uuid(doc_uuid)
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self._doc_selector(validated_uuid),
            )
            logger.info("Deleted Qdrant chunks for doc_uuid=%s", validated_uuid)
            return True
        except Exception as exc:
            logger.exception("Failed to delete Qdrant chunks for doc_uuid=%s", validated_uuid)
            raise QdrantSyncError(
                f"Failed to delete document from Qdrant for doc_uuid={validated_uuid}"
            ) from exc

    def update_document_metadata(self, doc_uuid: str, new_metadata: Dict[str, Any]) -> bool:
        validated_uuid = self.validate_doc_uuid(doc_uuid)
        safe_metadata = self.validate_metadata_update(new_metadata)

        if not safe_metadata:
            logger.info(
                "No metadata changes for doc_uuid=%s; skipping",
                validated_uuid,
            )
            return True

        payload = {f"metadata.{key}": value for key, value in safe_metadata.items()}

        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points_selector=self._doc_selector(validated_uuid),
            )
            logger.info(
                "Updated Qdrant metadata for doc_uuid=%s keys=%s",
                validated_uuid,
                sorted(safe_metadata.keys()),
            )
            return True
        except Exception as exc:
            logger.exception(
                "Failed to update Qdrant metadata for doc_uuid=%s", validated_uuid
            )
            raise QdrantSyncError(
                f"Failed to update metadata in Qdrant for doc_uuid={validated_uuid}"
            ) from exc

    def search_by_doc_uuid(self, doc_uuid: str, limit: int = 10) -> list:
        validated_uuid = self.validate_doc_uuid(doc_uuid)
        try:
            filter_condition = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.doc_uuid",
                        match=rest.MatchValue(value=validated_uuid),
                    )
                ]
            )
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
            )
            return results
        except Exception as exc:
            logger.error("Search by doc_uuid failed: %s", exc)
            raise

    def list_documents_by_department(self, department: str, limit: int = 20) -> list:
        normalized_dept = normalize_tag(department)
        if not normalized_dept:
            raise QdrantSyncError("department must not be empty")
        try:
            # prefer array field; also support legacy scalar via should
            filter_condition = rest.Filter(
                should=[
                    rest.FieldCondition(
                        key="metadata.departments",
                        match=rest.MatchAny(any=[normalized_dept]),
                    ),
                    rest.FieldCondition(
                        key="metadata.department",
                        match=rest.MatchValue(value=normalized_dept),
                    ),
                ],
                min_should=1,
            )
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
            )
            return results
        except Exception as exc:
            logger.error("List by department failed: %s", exc)
            raise


def get_qdrant_sync_manager() -> QdrantSyncManager:
    return QdrantSyncManager()


# Module-level singleton (existing imports: from auth_rbac import qdrant_sync)
qdrant_sync = get_qdrant_sync_manager()