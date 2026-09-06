# hybrid_retrieval.py
"""
Qdrant-native Hybrid retrieval: dense + sparse with RRF fusion + RBAC filter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models as rest

from auth_rbac import UserContext, RBACManager
from config import get_settings
from create_database import ensure_hybrid_collection
from get_embedding_function import get_embedding_function
from sparse_encoder import get_sparse_encoder

logger = logging.getLogger(__name__)


def _payload_to_doc(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not payload:
        return "", {}
    page = payload.get("page_content") or payload.get("text") or ""
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        meta = {k: v for k, v in payload.items() if k != "page_content"}
    return str(page), meta


def hybrid_search(
    query: str,
    user_context: UserContext,
    *,
    k: Optional[int] = None,
    query_filter: Optional[rest.Filter] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (org_context, sources) same shape as legacy RAGEngine retrieve.
    """
    settings = get_settings()
    k_final = int(k or settings.ai.retrieval_k)
    prefetch_limit = int(settings.ai.hybrid_prefetch_limit)
    dense_name = settings.ai.dense_vector_name
    sparse_name = settings.ai.sparse_vector_name
    collection = settings.db.qdrant_collection

    rbac = RBACManager()
    qfilter = query_filter if query_filter is not None else rbac.build_qdrant_filter(user_context)

    client = ensure_hybrid_collection()
    embedding = get_embedding_function()
    sparse_encoder = get_sparse_encoder()

    dense_q = embedding.embed_query(query)
    sparse_q = sparse_encoder.encode(query)

    if not settings.ai.hybrid_enabled:
        # Emergency dense-only on named vector
        results = client.search(
            collection_name=collection,
            query_vector=(dense_name, dense_q),
            query_filter=qfilter,
            limit=k_final,
            with_payload=True,
        )
        scored = results
    else:
        response = client.query_points(
            collection_name=collection,
            prefetch=[
                rest.Prefetch(
                    query=dense_q,
                    using=dense_name,
                    filter=qfilter,
                    limit=prefetch_limit,
                ),
                rest.Prefetch(
                    query=sparse_q,
                    using=sparse_name,
                    filter=qfilter,
                    limit=prefetch_limit,
                ),
            ],
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=k_final,
            with_payload=True,
        )
        scored = response.points

    sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []

    for point in scored:
        payload = point.payload or {}
        page_content, meta = _payload_to_doc(payload)
        score = float(point.score) if point.score is not None else 0.0
        if page_content:
            context_parts.append(page_content)
        sources.append(
            {
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index"),
                "doc_uuid": meta.get("doc_uuid"),
                "roles": meta.get("roles"),
                "departments": meta.get("departments"),
                "status": meta.get("status"),
                "relevance_score": score,
            }
        )

    org_context = "\n\n".join(context_parts)
    logger.info(
        "Hybrid retrieve | user=%s | hits=%s | hybrid=%s",
        getattr(user_context, "user_id", None),
        len(sources),
        settings.ai.hybrid_enabled,
    )
    return org_context, sources