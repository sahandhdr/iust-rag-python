# sparse_encoder.py
"""
Hashing-BM25-style sparse encoder for Qdrant named sparse vectors.

- Same encoder MUST be used at ingest and query time.
- No external service; pure Python; production-safe and testable.
- Indices are stable MD5-derived uints; values are TF-normalized weights.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import List, Tuple

from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

# Unicode letters/digits including Persian range
_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text))]


def _term_index(term: str) -> int:
    digest = hashlib.md5(term.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


class SparseEncoder:
    """
    Encode text to qdrant_client SparseVector.
    Empty text → tiny non-empty vector (Qdrant rejects empty sparse).
    """

    def encode(self, text: str) -> rest.SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return rest.SparseVector(indices=[0], values=[1e-8])

        tf = Counter(tokens)
        pairs: List[Tuple[int, float]] = []
        for term, count in tf.items():
            idx = _term_index(term)
            # simple TF saturation (BM25-like component without corpus DF)
            weight = float(count) / (1.0 + float(count))
            pairs.append((idx, weight))

        # Qdrant requires unique indices sorted ascending
        merged: dict[int, float] = {}
        for idx, weight in pairs:
            merged[idx] = merged.get(idx, 0.0) + weight

        ordered = sorted(merged.items(), key=lambda x: x[0])
        indices = [i for i, _ in ordered]
        values = [v for _, v in ordered]
        return rest.SparseVector(indices=indices, values=values)


_sparse_encoder: SparseEncoder | None = None


def get_sparse_encoder() -> SparseEncoder:
    global _sparse_encoder
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    return _sparse_encoder