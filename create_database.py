"""
Vector store layer — chunking + Qdrant persistence.

Metadata contract (every chunk payload):
  doc_uuid      str          required
  roles         list[str]    required (may be empty only for admin-only docs)
  departments   list[str]    required (may be empty)
  permissions   list[str]    optional
  status        str          required (default: published)
  version       int          required (default: 1)
  department    str          legacy scalar (first department or "public")
  source        str
  chunk_index   int
  total_chunks  int

Fail-closed: empty text / invalid tags → raise, never silent partial write.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config import get_settings
from get_embedding_function import get_embedding_function

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def normalize_tag(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("tag must be a string")
    return value.strip().lower()


def normalize_tag_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    result: Set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        tag = normalize_tag(item)
        if tag:
            result.add(tag)
    return sorted(result)


def build_document_metadata(
    *,
    doc_uuid: str,
    source: str,
    chunk_index: int,
    total_chunks: int,
    roles: Optional[Sequence[str]] = None,
    departments: Optional[Sequence[str]] = None,
    permissions: Optional[Sequence[str]] = None,
    status: Optional[str] = None,
    version: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build normalized metadata dict for a single chunk.
    Always writes both multi-tag fields and legacy scalar `department`.
    """
    settings = get_settings()

    roles_n = normalize_tag_list(roles)
    depts_n = normalize_tag_list(departments)
    perms_n = normalize_tag_list(permissions)

    if not roles_n:
        roles_n = normalize_tag_list(settings.ingestion.default_roles) or ["public"]

    status_n = normalize_tag(status) if status else normalize_tag(
        settings.ingestion.default_status
    )
    if not status_n:
        status_n = "published"

    if not isinstance(version, int) or version < 1:
        raise ValueError("version must be a positive integer")

    # legacy scalar: first department, else "public" if public role, else first role
    if depts_n:
        legacy_department = depts_n[0]
    elif "public" in roles_n:
        legacy_department = "public"
    else:
        legacy_department = roles_n[0] if roles_n else "public"

    meta: Dict[str, Any] = {
        "doc_uuid": doc_uuid.strip(),
        "roles": roles_n,
        "departments": depts_n,
        "permissions": perms_n,
        "status": status_n,
        "version": version,
        "department": legacy_department,
        "source": source,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if extra:
        for key, value in extra.items():
            if key in meta:
                continue
            meta[key] = value
    return meta


# ---------------------------------------------------------------------------
# Qdrant client / collection
# ---------------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    qdrant_path = settings.db.qdrant_path
    qdrant_url = (settings.db.qdrant_url or "").strip()

    try:
        if qdrant_url:
            logger.info("Connecting to remote Qdrant: %s", qdrant_url)
            return QdrantClient(url=qdrant_url)
        os.makedirs(qdrant_path, exist_ok=True)
        logger.info("Using local Qdrant path: %s", qdrant_path)
        return QdrantClient(path=qdrant_path)
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc, exc_info=True)
        raise


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1536,
) -> None:
    """
    Idempotent collection create.
    Default vector_size=1536 matches text-embedding-3-small;
    override when using a different embedding model.
    """
    try:
        if client.collection_exists(collection_name=collection_name):
            return
        logger.info("Creating collection '%s' (dim=%s)", collection_name, vector_size)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(
                size=vector_size,
                distance=rest_models.Distance.COSINE,
            ),
        )
        # payload indexes for ACL filters (best-effort; ignore if unsupported)
        for field_name, field_schema in (
            ("metadata.roles", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.departments", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.permissions", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.status", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.department", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.doc_uuid", rest_models.PayloadSchemaType.KEYWORD),
        ):
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as idx_exc:
                logger.debug("Payload index %s skipped: %s", field_name, idx_exc)
        logger.info("Collection '%s' ready", collection_name)
    except Exception as exc:
        logger.error("ensure_collection_exists failed: %s", exc, exc_info=True)
        raise


def _resolve_vector_size(embedding: Any) -> int:
    """Best-effort detect embedding dimension."""
    try:
        probe = embedding.embed_query("dimension probe")
        if isinstance(probe, list) and probe:
            return len(probe)
    except Exception as exc:
        logger.warning("Could not probe embedding dim; defaulting to 1536. err=%s", exc)
    return 1536


def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    collection_name = settings.db.qdrant_collection
    embedding = get_embedding_function()
    client = get_qdrant_client()
    ensure_collection_exists(
        client,
        collection_name,
        vector_size=_resolve_vector_size(embedding),
    )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Core ingest into Qdrant
# ---------------------------------------------------------------------------

def process_single_document(
    md_path: str,
    *,
    doc_uuid: Optional[str] = None,
    roles: Optional[Sequence[str]] = None,
    departments: Optional[Sequence[str]] = None,
    permissions: Optional[Sequence[str]] = None,
    status: Optional[str] = None,
    version: int = 1,
    overwrite: bool = True,
    # backward-compatible single department argument
    department: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chunk markdown file and upsert into Qdrant with full ACL metadata.

    If `department` is passed (legacy), it is merged into `departments`.
    """
    settings = get_settings()

    if not doc_uuid or not str(doc_uuid).strip():
        doc_uuid = str(uuid.uuid4())
        logger.info("Generated doc_uuid=%s for %s", doc_uuid, md_path)
    else:
        doc_uuid = str(doc_uuid).strip()
        logger.info("Using provided doc_uuid=%s", doc_uuid)

    # merge legacy department into departments list
    dept_list = list(departments or [])
    if department:
        dept_list.append(department)

    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if not text or not text.strip():
        raise ValueError(f"No content in {md_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.ingestion.chunk_size,
        chunk_overlap=settings.ingestion.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError(f"Splitter produced zero chunks for {md_path}")

    source_name = os.path.basename(md_path)
    documents: List[Document] = []
    for index, chunk in enumerate(chunks):
        meta = build_document_metadata(
            doc_uuid=doc_uuid,
            source=source_name,
            chunk_index=index,
            total_chunks=len(chunks),
            roles=roles,
            departments=dept_list,
            permissions=permissions,
            status=status,
            version=version,
        )
        documents.append(Document(page_content=chunk, metadata=meta))

    if overwrite:
        from auth_rbac import get_qdrant_sync_manager

        sync_manager = get_qdrant_sync_manager()
        try:
            sync_manager.delete_document_by_uuid(doc_uuid)
            logger.info("Overwrite: deleted previous chunks for doc_uuid=%s", doc_uuid)
        except Exception as del_exc:
            # first-time insert: delete may fail if nothing exists — log and continue
            logger.warning(
                "Overwrite delete skipped/failed for doc_uuid=%s: %s",
                doc_uuid,
                del_exc,
            )

    vector_store = get_vector_store()
    vector_store.add_documents(documents)

    first_meta = documents[0].metadata
    logger.info(
        "Ingested %s chunks | doc_uuid=%s | roles=%s | departments=%s | status=%s | file=%s",
        len(chunks),
        doc_uuid,
        first_meta.get("roles"),
        first_meta.get("departments"),
        first_meta.get("status"),
        source_name,
    )
    return {
        "status": "success",
        "doc_uuid": doc_uuid,
        "chunks": len(chunks),
        "roles": first_meta.get("roles"),
        "departments": first_meta.get("departments"),
        "permissions": first_meta.get("permissions"),
        "status": first_meta.get("status"),
        "version": first_meta.get("version"),
        "file": source_name,
    }


def delete_document_from_qdrant(doc_uuid: str) -> bool:
    try:
        from auth_rbac import get_qdrant_sync_manager

        return get_qdrant_sync_manager().delete_document_by_uuid(doc_uuid)
    except Exception as exc:
        logger.error("delete_document_from_qdrant failed doc_uuid=%s: %s", doc_uuid, exc)
        return False


def update_document_metadata_in_qdrant(
    doc_uuid: str,
    new_metadata: Dict[str, Any],
) -> bool:
    try:
        from auth_rbac import get_qdrant_sync_manager

        return get_qdrant_sync_manager().update_document_metadata(doc_uuid, new_metadata)
    except Exception as exc:
        logger.error(
            "update_document_metadata_in_qdrant failed doc_uuid=%s: %s",
            doc_uuid,
            exc,
        )
        return False


__all__ = [
    "normalize_tag",
    "normalize_tag_list",
    "build_document_metadata",
    "get_qdrant_client",
    "ensure_collection_exists",
    "get_vector_store",
    "process_single_document",
    "delete_document_from_qdrant",
    "update_document_metadata_in_qdrant",
]