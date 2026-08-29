"""
Unit tests for DocumentAccessPolicy (or / and / hybrid).

Run:
  cd Version_4 && python -m pytest tests/test_document_access_policy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from access_control.policy import (
    AccessMode,
    HybridBothMode,
    DocumentAccessConfig,
    DocumentAccessPolicy,
)


def _user(roles=None, departments=None, permissions=None, user_id: int = 10):
    return {
        "user_id": user_id,
        "username": "tester",
        "roles": set(roles or ["public"]),
        "departments": set(departments or []),
        "permissions": set(permissions or []),
    }


def _policy(mode: str = "or", hybrid_both: str = "or", enable_perm: bool = False):
    cfg = DocumentAccessConfig(
        access_mode=AccessMode(mode),
        hybrid_both_mode=HybridBothMode(hybrid_both),
        enable_permission_tag_match=enable_perm,
        require_published=True,
    )
    return DocumentAccessPolicy(cfg)


class TestOrMode:
    def test_public_role_sees_public_doc(self):
        p = _policy("or")
        u = _user(roles=["public"])
        assert p.evaluate_document_access(
            u, doc_roles=["public"], doc_departments=[], doc_status="published"
        )

    def test_dept_match_without_role(self):
        p = _policy("or")
        u = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u, doc_roles=["manager"], doc_departments=["it"], doc_status="published"
        )

    def test_no_match_denied(self):
        p = _policy("or")
        u = _user(roles=["staff"], departments=["hr"])
        assert not p.evaluate_document_access(
            u, doc_roles=["manager"], doc_departments=["it"], doc_status="published"
        )

    def test_draft_denied(self):
        p = _policy("or")
        u = _user(roles=["public"])
        assert not p.evaluate_document_access(
            u, doc_roles=["public"], doc_status="draft"
        )


class TestAndMode:
    def test_both_required_when_doc_has_both(self):
        p = _policy("and")
        u = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=["it"], doc_status="published"
        )
        assert not p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=["hr"], doc_status="published"
        )

    def test_single_dimension_role_only_doc(self):
        p = _policy("and")
        u = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=[], doc_status="published"
        )

    def test_single_dimension_dept_only_doc(self):
        p = _policy("and")
        u = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u, doc_roles=[], doc_departments=["it"], doc_status="published"
        )


class TestHybridMode:
    def test_role_only_doc(self):
        p = _policy("hybrid", hybrid_both="or")
        u = _user(roles=["staff"], departments=["hr"])
        assert p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=[], doc_status="published"
        )

    def test_dept_only_doc(self):
        p = _policy("hybrid", hybrid_both="or")
        u = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u, doc_roles=[], doc_departments=["it"], doc_status="published"
        )

    def test_both_tags_hybrid_or(self):
        p = _policy("hybrid", hybrid_both="or")
        u = _user(roles=["staff"], departments=["hr"])
        assert p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=["it"], doc_status="published"
        )

    def test_both_tags_hybrid_and(self):
        p = _policy("hybrid", hybrid_both="and")
        u = _user(roles=["staff"], departments=["hr"])
        assert not p.evaluate_document_access(
            u, doc_roles=["staff"], doc_departments=["it"], doc_status="published"
        )
        u2 = _user(roles=["staff"], departments=["it"])
        assert p.evaluate_document_access(
            u2, doc_roles=["staff"], doc_departments=["it"], doc_status="published"
        )

    def test_untagged_doc_denied(self):
        p = _policy("hybrid")
        u = _user(roles=["staff"], departments=["it"])
        assert not p.evaluate_document_access(
            u, doc_roles=[], doc_departments=[], doc_status="published"
        )


class TestAdminBypass:
    def test_admin_with_bypass_perm(self):
        p = _policy("and")
        u = _user(roles=["admin"], permissions=["rbac.bypass"], departments=[])
        assert p.evaluate_document_access(
            u, doc_roles=["manager"], doc_departments=["secret"], doc_status="published"
        )

    def test_admin_without_bypass_perm_no_special(self):
        p = _policy("or")
        u = _user(roles=["admin"], permissions=[], departments=[])
        assert not p.evaluate_document_access(
            u, doc_roles=["manager"], doc_departments=["secret"], doc_status="published"
        )


class TestFailClosed:
    def test_empty_roles_raises(self):
        p = _policy("or")
        with pytest.raises(ValueError):
            p.evaluate_document_access(
                {"user_id": 1, "roles": [], "departments": [], "permissions": []},
                doc_roles=["public"],
                doc_status="published",
            )

    def test_build_filter_requires_qdrant(self):
        p = _policy("or")
        u = _user(roles=["public"])
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError):
                p.build_qdrant_filter(u)
            return
        f = p.build_qdrant_filter(u)
        assert f is not None