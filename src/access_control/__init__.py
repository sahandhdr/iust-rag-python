"""
Document Access Control package.

Public API:
- DocumentAccessPolicy: builds Qdrant filters (or / and / hybrid)
- AccessMode, HybridBothMode: enums for config
- normalize_tag / normalize_tags: shared normalization
"""

from .policy import (
    AccessMode,
    HybridBothMode,
    DocumentAccessPolicy,
    DocumentAccessConfig,
    normalize_tag,
    normalize_tags,
)

__all__ = [
    "AccessMode",
    "HybridBothMode",
    "DocumentAccessPolicy",
    "DocumentAccessConfig",
    "normalize_tag",
    "normalize_tags",
]