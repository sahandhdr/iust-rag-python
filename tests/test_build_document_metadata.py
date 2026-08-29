"""
Unit tests for access_control.metadata (build_document_metadata + tag helpers).

Run:
  cd Version_4
  python -m pytest tests/test_build_document_metadata.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ..access_control.metadata import (
    build_document_metadata,
    normalize_tag,
    normalize_tag_list,
)


class TestNormalizeTag:
    def test_strip_lower(self):
        assert normalize_tag("  Public ") == "public"

    def test_none(self):
        assert normalize_tag(None) == ""

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            normalize_tag(123)  # type: ignore[arg-type]


class TestNormalizeTagList:
    def test_list_and_dedupe(self):
        assert normalize_tag_list(["IT", "it", " HR "]) == ["hr", "it"]

    def test_string_input(self):
        assert normalize_tag_list("Public") == ["public"]

    def test_none(self):
        assert normalize_tag_list(None) == []


class TestBuildDocumentMetadata:
    def test_full_tags(self):
        meta = build_document_metadata(
            doc_uuid="abc-123",
            source="policy.md",
            chunk_index=0,
            total_chunks=3,
            roles=["Public", "staff"],
            departments=["IT"],
            permissions=["documents.read.public"],
            status="Published",
            version=2,
        )
        assert meta["doc_uuid"] == "abc-123"
        assert meta["roles"] == ["public", "staff"]
        assert meta["departments"] == ["it"]
        assert meta["permissions"] == ["documents.read.public"]
        assert meta["status"] == "published"
        assert meta["version"] == 2
        assert meta["department"] == "it"
        assert meta["source"] == "policy.md"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 3

    def test_default_role_when_empty(self):
        meta = build_document_metadata(
            doc_uuid="u1",
            source="a.md",
            chunk_index=0,
            total_chunks=1,
            roles=[],
            departments=[],
            default_roles=["public"],
            default_status="published",
        )
        assert "public" in meta["roles"]
        assert meta["status"] == "published"
        assert meta["department"] == "public"

    def test_legacy_department_from_role_when_no_dept(self):
        meta = build_document_metadata(
            doc_uuid="u2",
            source="b.md",
            chunk_index=0,
            total_chunks=1,
            roles=["staff"],
            departments=[],
        )
        assert meta["department"] == "staff"
        assert meta["roles"] == ["staff"]
        assert meta["departments"] == []

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError):
            build_document_metadata(
                doc_uuid="u3",
                source="c.md",
                chunk_index=0,
                total_chunks=1,
                version=0,
            )

    def test_extra_fields_merged(self):
        meta = build_document_metadata(
            doc_uuid="u4",
            source="d.md",
            chunk_index=1,
            total_chunks=2,
            roles=["public"],
            extra={"page": 5, "custom": "x"},
        )
        assert meta["page"] == 5
        assert meta["custom"] == "x"
        assert meta["roles"] == ["public"]