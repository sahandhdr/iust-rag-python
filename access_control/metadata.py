"""
Document chunk metadata builders (no Qdrant / LangChain dependency).

Used by create_database.process_single_document and unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


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
    default_roles: Optional[Sequence[str]] = None,
    default_status: str = "published",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build normalized metadata dict for a single chunk.
    Always writes multi-tag fields and legacy scalar `department`.
    """
    roles_n = normalize_tag_list(roles)
    depts_n = normalize_tag_list(departments)
    perms_n = normalize_tag_list(permissions)

    if not roles_n:
        roles_n = normalize_tag_list(default_roles) or ["public"]

    status_n = normalize_tag(status) if status else normalize_tag(default_status)
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