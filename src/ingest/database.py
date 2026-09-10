# database.py

# """
# Vector store layer — chunking + Qdrant persistence.
#
# Metadata contract (every chunk payload):
#   doc_uuid      str          required
#   roles         list[str]    required (may be empty only for admin-only docs)
#   departments   list[str]    required (may be empty)
#   permissions   list[str]    optional
#   status        str          required (default: published)
#   version       int          required (default: 1)
#   department    str          legacy scalar (first department or "public")
#   source        str
#   chunk_index   int
#   total_chunks  int
#
# Fail-closed: empty text / invalid tags → raise, never silent partial write.
# """
"""
# Vector store layer — chunking + Qdrant persistence with Hybrid (dense + sparse).
#
# Point layout:
#   vectors:  { dense: [...], sparse: SparseVector }
#   payload:  { page_content, metadata: { doc_uuid, roles, ... } }
#
# Collection is created from code only (no manual Qdrant UI steps).
# """

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from config.settings import get_settings
from models.embeddings import get_embedding_function
from core.sparse_encoder import get_sparse_encoder

logger = logging.getLogger(__name__)


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


def _resolve_vector_size(embedding: Any) -> int:
    try:
        probe = embedding.embed_query("dimension probe")
        if isinstance(probe, list) and probe:
            return len(probe)
    except Exception as exc:
        logger.warning("Could not probe embedding dim; defaulting to 1536. err=%s", exc)
    return 1536


def _count_points(client: QdrantClient, collection_name: str) -> int:
    try:
        if not client.collection_exists(collection_name=collection_name):
            return 0
        info = client.get_collection(collection_name=collection_name)
        return int(getattr(info, "points_count", None) or 0)
    except Exception as exc:
        logger.warning("points_count failed for %s: %s", collection_name, exc)
        return 0


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1536,
) -> None:
    """
    Idempotent create for Hybrid collection:
      dense  — COSINE dense vector
      sparse — named sparse vector
    If collection already exists, leave it (ops must use wipe to reset).
    """
    settings = get_settings()
    dense_name = settings.ai.dense_vector_name
    sparse_name = settings.ai.sparse_vector_name

    try:
        if client.collection_exists(collection_name=collection_name):
            logger.info("Collection '%s' already exists — skipping create", collection_name)
            return

        logger.info(
            "Creating hybrid collection '%s' (dense_dim=%s, dense=%s, sparse=%s)",
            collection_name,
            vector_size,
            dense_name,
            sparse_name,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_name: rest_models.VectorParams(
                    size=vector_size,
                    distance=rest_models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                sparse_name: rest_models.SparseVectorParams(
                    index=rest_models.SparseIndexParams(
                        on_disk=False,
                    ),
                ),
            },
        )

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

        logger.info("Hybrid collection '%s' ready", collection_name)
    except Exception as exc:
        logger.error("ensure_collection_exists failed: %s", exc, exc_info=True)
        raise


def ensure_hybrid_collection() -> QdrantClient:
    """Connect + ensure hybrid schema. Call from ingest/retrieve."""
    settings = get_settings()
    client = get_qdrant_client()
    embedding = get_embedding_function()
    ensure_collection_exists(
        client,
        settings.db.qdrant_collection,
        vector_size=_resolve_vector_size(embedding),
    )
    return client


def wipe_and_recreate_collection() -> Dict[str, Any]:
    """
    Production wipe:
      1) count points (best-effort)
      2) delete collection if exists
      3) recreate Hybrid schema via ensure_collection_exists
    Does NOT touch MySQL or Laravel disk. Does NOT purge data/ orphans (P1-29).
    Idempotent if collection was already missing.
    """
    settings = get_settings()
    collection = settings.db.qdrant_collection
    client = get_qdrant_client()
    embedding = get_embedding_function()
    vector_size = _resolve_vector_size(embedding)

    points_before = _count_points(client, collection)
    existed = False
    try:
        existed = bool(client.collection_exists(collection_name=collection))
    except Exception as exc:
        logger.warning("collection_exists check failed: %s", exc)

    wiped = False
    if existed:
        try:
            client.delete_collection(collection_name=collection)
            wiped = True
            logger.info("Deleted Qdrant collection '%s' (points_before≈%s)", collection, points_before)
        except Exception as exc:
            logger.error("delete_collection failed: %s", exc, exc_info=True)
            raise

    ensure_collection_exists(client, collection, vector_size=vector_size)
    points_after = _count_points(client, collection)

    return {
        "collection": collection,
        "wiped": wiped,
        "recreated": True,
        "hybrid": True,
        "points_before": points_before,
        "points_after": points_after,
        "dense_vector": settings.ai.dense_vector_name,
        "sparse_vector": settings.ai.sparse_vector_name,
        "vector_size": vector_size,
    }


def delete_document_markdown_files(doc_uuid: str) -> int:
    """Remove derivative markdown files data/**/{doc_uuid}.md. Returns count deleted."""
    settings = get_settings()
    data_dir = settings.ingestion.data_dir
    if not doc_uuid or not str(doc_uuid).strip():
        return 0
    safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", str(doc_uuid).strip())
    target_names = {f"{safe}.md"}
    removed = 0
    if not os.path.isdir(data_dir):
        return 0
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            if name in target_names:
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed += 1
                    logger.info("Removed derivative markdown: %s", path)
                except OSError as exc:
                    logger.warning("Failed to remove %s: %s", path, exc)
    return removed

def _safe_doc_uuid_filename(doc_uuid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-_]", "_", str(doc_uuid).strip())


def _qdrant_has_doc_uuid(client: QdrantClient, collection_name: str, doc_uuid: str) -> bool:
    """True if at least one point exists for metadata.doc_uuid."""
    try:
        if not client.collection_exists(collection_name=collection_name):
            return False
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="metadata.doc_uuid",
                        match=rest_models.MatchValue(value=doc_uuid),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
    except Exception as exc:
        logger.warning("scroll check failed for doc_uuid=%s: %s", doc_uuid, exc)
        return True  # fail-closed: do not delete if uncertain


def cleanup_orphan_data_files(
    *,
    dry_run: bool = True,
    remove_empty_dirs: bool = True,
    cleanup_temp_ingest: bool = True,
) -> Dict[str, Any]:
    """
    Remove derivative markdown under data_dir when no Qdrant points exist for that doc_uuid.
    Optionally remove cwd temp_ingest_* leftovers from direct API tests.
    Never touches Laravel storage or Qdrant collection itself.
    """
    settings = get_settings()
    data_dir = settings.ingestion.data_dir
    collection_name = settings.db.qdrant_collection
    client = get_qdrant_client()

    scanned = 0
    orphans: List[Dict[str, Any]] = []
    kept = 0
    removed_files = 0
    removed_dirs = 0
    temp_removed = 0
    errors: List[str] = []

    if os.path.isdir(data_dir):
        for root, _dirs, files in os.walk(data_dir):
            for name in files:
                if not name.lower().endswith(".md"):
                    continue
                scanned += 1
                path = os.path.join(root, name)
                stem = name[:-3]  # strip .md
                # stem is safe filename form of doc_uuid
                in_qdrant = _qdrant_has_doc_uuid(client, collection_name, stem)
                # also try original if identical
                if not in_qdrant and stem != name:
                    in_qdrant = _qdrant_has_doc_uuid(client, collection_name, stem)

                if in_qdrant:
                    kept += 1
                    continue

                entry = {"path": path, "doc_uuid_key": stem, "reason": "not-in-qdrant"}
                orphans.append(entry)
                if dry_run:
                    continue
                try:
                    os.remove(path)
                    removed_files += 1
                    logger.info("Orphan markdown removed: %s", path)
                except OSError as exc:
                    errors.append(f"{path}: {exc}")

        if remove_empty_dirs and not dry_run:
            for root, dirs, files in os.walk(data_dir, topdown=False):
                if root == os.path.abspath(data_dir) or root == data_dir:
                    continue
                if not dirs and not files:
                    try:
                        os.rmdir(root)
                        removed_dirs += 1
                    except OSError:
                        pass

    if cleanup_temp_ingest:
        cwd = os.getcwd()
        try:
            for name in os.listdir(cwd):
                if not name.startswith("temp_ingest_"):
                    continue
                path = os.path.join(cwd, name)
                if not os.path.isfile(path):
                    continue
                orphans.append({"path": path, "doc_uuid_key": None, "reason": "temp_ingest"})
                if dry_run:
                    continue
                try:
                    os.remove(path)
                    temp_removed += 1
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
        except OSError as exc:
            errors.append(f"cwd-scan: {exc}")

    return {
        "data_dir": os.path.abspath(data_dir),
        "dry_run": dry_run,
        "scanned_md": scanned,
        "kept_in_qdrant": kept,
        "orphan_candidates": len([o for o in orphans if o.get("reason") == "not-in-qdrant"]),
        "orphans": orphans[:200],
        "removed_files": removed_files,
        "removed_empty_dirs": removed_dirs,
        "temp_ingest_removed": temp_removed,
        "errors": errors,
    }

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
    department: Optional[str] = None,
) -> Dict[str, Any]:
    """Chunk markdown and upsert into Qdrant with dense + sparse vectors."""
    settings = get_settings()
    dense_name = settings.ai.dense_vector_name
    sparse_name = settings.ai.sparse_vector_name

    if not doc_uuid or not str(doc_uuid).strip():
        doc_uuid = str(uuid.uuid4())
        logger.info("Generated doc_uuid=%s for %s", doc_uuid, md_path)
    else:
        doc_uuid = str(doc_uuid).strip()
        logger.info("Using provided doc_uuid=%s", doc_uuid)

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

    if overwrite:
        try:
            delete_document_from_qdrant(doc_uuid, delete_markdown=False)
            logger.info("Overwrite: deleted previous Qdrant points for doc_uuid=%s", doc_uuid)
        except Exception as del_exc:
            logger.warning(
                "Overwrite delete skipped/failed for doc_uuid=%s: %s",
                doc_uuid,
                del_exc,
            )

    client = ensure_hybrid_collection()
    embedding = get_embedding_function()
    sparse_encoder = get_sparse_encoder()
    source_name = os.path.basename(md_path)

    dense_vectors = embedding.embed_documents(list(chunks))
    if len(dense_vectors) != len(chunks):
        raise RuntimeError("embed_documents size mismatch")

    points: List[rest_models.PointStruct] = []
    first_meta: Dict[str, Any] = {}

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
        if index == 0:
            first_meta = meta

        point_id = str(uuid.uuid4())
        sparse_vec = sparse_encoder.encode(chunk)
        points.append(
            rest_models.PointStruct(
                id=point_id,
                vector={
                    dense_name: dense_vectors[index],
                    sparse_name: sparse_vec,
                },
                payload={
                    "page_content": chunk,
                    "metadata": meta,
                },
            )
        )

    client.upsert(
        collection_name=settings.db.qdrant_collection,
        points=points,
        wait=True,
    )

    logger.info(
        "Hybrid ingested %s chunks | doc_uuid=%s | roles=%s | departments=%s | status=%s | file=%s",
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
        "hybrid": True,
    }


def delete_document_from_qdrant(doc_uuid: str, delete_markdown: bool = True) -> bool:
    """Delete all points for doc_uuid; optionally remove derivative .md files."""
    if not doc_uuid or not str(doc_uuid).strip():
        return False
    validated = str(doc_uuid).strip()
    settings = get_settings()
    client = get_qdrant_client()

    try:
        if client.collection_exists(collection_name=settings.db.qdrant_collection):
            client.delete(
                collection_name=settings.db.qdrant_collection,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="metadata.doc_uuid",
                                match=rest_models.MatchValue(value=validated),
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted Qdrant hybrid points for doc_uuid=%s", validated)
        else:
            logger.warning(
                "Collection %s missing on delete doc_uuid=%s",
                settings.db.qdrant_collection,
                validated,
            )
    except Exception as exc:
        logger.error("delete_document_from_qdrant failed doc_uuid=%s: %s", validated, exc)
        return False

    if delete_markdown:
        delete_document_markdown_files(validated)
    return True


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


def get_vector_store():
    """Legacy helper. Ensures collection exists and returns client."""
    return ensure_hybrid_collection()


__all__ = [
    "normalize_tag",
    "normalize_tag_list",
    "build_document_metadata",
    "get_qdrant_client",
    "ensure_collection_exists",
    "ensure_hybrid_collection",
    "wipe_and_recreate_collection",
    "get_vector_store",
    "process_single_document",
    "delete_document_from_qdrant",
    "delete_document_markdown_files",
    "update_document_metadata_in_qdrant",
    "cleanup_orphan_data_files",
]