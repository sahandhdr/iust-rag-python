"""
Document Access Policy — production-ready, config-driven.

Modes:
  - or:      role match OR department match (optional permission match)
  - and:     role match AND department match (with single-dimension fallback)
  - hybrid:  single-tag docs → that dimension only;
             both-tag docs → HYBRID_BOTH_MODE (or | and)

Fail-closed: invalid context / empty effective filters → impossible filter.

Compatibility note:
  qdrant-client (recent versions) requires min_should to be a MinShould object,
  not a plain int.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Union

try:
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover
    rest = None  # type: ignore

logger = logging.getLogger(__name__)


class AccessMode(str, Enum):
    OR = "or"
    AND = "and"
    HYBRID = "hybrid"


class HybridBothMode(str, Enum):
    OR = "or"
    AND = "and"


def normalize_tag(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("tag value must be a string")
    return value.strip().lower()


def normalize_tags(values: Optional[Iterable[Any]]) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    result: Set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        normalized = normalize_tag(item)
        if normalized:
            result.add(normalized)
    return result


def _make_min_should(conditions: List[Any], min_count: int = 1) -> Any:
    """
    Create a proper MinShould object compatible with recent qdrant-client.
    Falls back gracefully if MinShould is unavailable (very old client).
    """
    if rest is None:
        return None
    if hasattr(rest, "MinShould"):
        return rest.MinShould(conditions=conditions, min_count=min_count)
    # Extremely old client fallback (should not happen in current stack)
    return min_count


@dataclass(frozen=True)
class DocumentAccessConfig:
    access_mode: AccessMode = AccessMode.OR
    hybrid_both_mode: HybridBothMode = HybridBothMode.OR
    admin_roles: frozenset[str] = frozenset({"admin", "developer", "superadmin"})
    admin_bypass_permissions: frozenset[str] = frozenset(
        {"all", "documents.read.all", "rbac.bypass"}
    )
    require_published: bool = True
    published_status_value: str = "published"
    enable_permission_tag_match: bool = False
    roles_field: str = "metadata.roles"
    departments_field: str = "metadata.departments"
    permissions_field: str = "metadata.permissions"
    status_field: str = "metadata.status"
    legacy_department_field: str = "metadata.department"
    support_legacy_department: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "DocumentAccessConfig":
        rbac = getattr(settings, "rbac", None)

        def _get(name: str, default: Any) -> Any:
            if rbac is None:
                return default
            return getattr(rbac, name, default)

        mode_raw = str(_get("access_mode", "or")).strip().lower()
        hybrid_raw = str(_get("hybrid_both_mode", "or")).strip().lower()

        try:
            access_mode = AccessMode(mode_raw)
        except ValueError:
            logger.warning("Invalid access_mode=%s; falling back to 'or'", mode_raw)
            access_mode = AccessMode.OR

        try:
            hybrid_both = HybridBothMode(hybrid_raw)
        except ValueError:
            logger.warning("Invalid hybrid_both_mode=%s; falling back to 'or'", hybrid_raw)
            hybrid_both = HybridBothMode.OR

        admin_roles = normalize_tags(
            _get("admin_roles", ["admin", "developer", "superadmin"])
        )
        bypass_perms = normalize_tags(
            _get(
                "admin_bypass_permissions",
                ["all", "documents.read.all", "rbac.bypass"],
            )
        )

        return cls(
            access_mode=access_mode,
            hybrid_both_mode=hybrid_both,
            admin_roles=frozenset(admin_roles),
            admin_bypass_permissions=frozenset(bypass_perms),
            require_published=bool(_get("require_published", True)),
            published_status_value=normalize_tag(
                str(_get("published_status_value", "published"))
            )
            or "published",
            enable_permission_tag_match=bool(
                _get("enable_permission_tag_match", False)
            ),
            support_legacy_department=bool(_get("support_legacy_department", True)),
        )


def _extract_user_sets(user_context: Any) -> Dict[str, Set[str]]:
    if user_context is None:
        raise ValueError("user_context is required")

    if isinstance(user_context, dict):
        roles = normalize_tags(user_context.get("roles"))
        departments = normalize_tags(user_context.get("departments"))
        permissions = normalize_tags(user_context.get("permissions"))
        user_id = user_context.get("user_id")
    else:
        roles = normalize_tags(getattr(user_context, "roles", None))
        departments = normalize_tags(getattr(user_context, "departments", None))
        permissions = normalize_tags(getattr(user_context, "permissions", None))
        user_id = getattr(user_context, "user_id", None)

    if not roles:
        raise ValueError("user_context.roles must not be empty (fail-closed)")

    if user_id is None or (isinstance(user_id, int) and user_id <= 0):
        raise ValueError("user_context.user_id must be a positive integer")

    return {
        "roles": roles,
        "departments": departments,
        "permissions": permissions,
    }


class DocumentAccessPolicy:
    FORBIDDEN_VALUE = "__forbidden__"

    def __init__(self, config: Optional[DocumentAccessConfig] = None):
        self.config = config or DocumentAccessConfig()

    @classmethod
    def from_settings(cls, settings: Any) -> "DocumentAccessPolicy":
        return cls(DocumentAccessConfig.from_settings(settings))

    def is_admin(self, user_context: Any) -> bool:
        sets = _extract_user_sets(user_context)
        return bool(sets["roles"] & set(self.config.admin_roles))

    def can_bypass(self, user_context: Any) -> bool:
        sets = _extract_user_sets(user_context)
        if not (sets["roles"] & set(self.config.admin_roles)):
            return False
        perms = sets["permissions"]
        if "all" in perms:
            return True
        return bool(perms & set(self.config.admin_bypass_permissions))

    def evaluate_document_access(
        self,
        user_context: Any,
        doc_roles: Optional[Iterable[str]] = None,
        doc_departments: Optional[Iterable[str]] = None,
        doc_permissions: Optional[Iterable[str]] = None,
        doc_status: Optional[str] = None,
    ) -> bool:
        if self.can_bypass(user_context):
            if self.config.require_published and doc_status is not None:
                return normalize_tag(doc_status) == self.config.published_status_value
            return True

        if self.config.require_published and doc_status is not None:
            if normalize_tag(doc_status) != self.config.published_status_value:
                return False

        sets = _extract_user_sets(user_context)
        d_roles = normalize_tags(doc_roles)
        d_depts = normalize_tags(doc_departments)
        d_perms = normalize_tags(doc_permissions)

        role_match = bool(sets["roles"] & d_roles)
        dept_match = bool(sets["departments"] & d_depts)
        perm_match = (
            bool(sets["permissions"] & d_perms)
            if self.config.enable_permission_tag_match
            else False
        )

        has_roles = bool(d_roles)
        has_depts = bool(d_depts)
        mode = self.config.access_mode

        if mode == AccessMode.OR:
            return role_match or dept_match or perm_match

        if mode == AccessMode.AND:
            if has_roles and has_depts:
                base = role_match and dept_match
            elif has_roles:
                base = role_match
            elif has_depts:
                base = dept_match
            else:
                base = False
            return base or perm_match

        if mode == AccessMode.HYBRID:
            if has_roles and not has_depts:
                return role_match or perm_match
            if has_depts and not has_roles:
                return dept_match or perm_match
            if has_roles and has_depts:
                if self.config.hybrid_both_mode == HybridBothMode.AND:
                    return (role_match and dept_match) or perm_match
                return role_match or dept_match or perm_match
            return False

        return False

    def build_qdrant_filter(self, user_context: Any) -> Optional[Any]:
        if rest is None:
            raise RuntimeError(
                "qdrant_client is not installed; cannot build Qdrant filters."
            )

        try:
            if self.can_bypass(user_context):
                if self.config.require_published:
                    return rest.Filter(must=[self._status_condition()])
                return None

            sets = _extract_user_sets(user_context)
            mode = self.config.access_mode

            must: List[Any] = []
            if self.config.require_published:
                must.append(self._status_condition())

            acl = self._build_acl_conditions(sets, mode)
            if acl is None:
                return self._forbidden_filter(must)

            if isinstance(acl, rest.Filter):
                # Merge must conditions
                if acl.must:
                    must.extend(list(acl.must))

                # Handle both old-style (should) and new-style (min_should)
                min_should_obj = getattr(acl, "min_should", None)
                should_list = list(acl.should or [])

                if min_should_obj is not None:
                    # New style: MinShould object already built
                    return rest.Filter(
                        must=must or None,
                        min_should=min_should_obj,
                    )

                if should_list:
                    # Old style fallback
                    return rest.Filter(
                        must=must or None,
                        min_should=_make_min_should(should_list, min_count=1),
                    )

                if not must:
                    return self._forbidden_filter([])
                return rest.Filter(must=must)

            # Single condition
            must.append(acl)
            return rest.Filter(must=must)

        except Exception as exc:
            logger.exception(
                "Failed to build access filter; applying fail-closed. err=%s", exc
            )
            return self._forbidden_filter(
                [self._status_condition()] if self.config.require_published else []
            )

    def _build_acl_conditions(
        self, sets: Dict[str, Set[str]], mode: AccessMode
    ) -> Optional[Any]:
        role_cond = self._overlap_condition(self.config.roles_field, sets["roles"])
        dept_parts = self._department_conditions(sets["departments"])
        perm_cond = None
        if self.config.enable_permission_tag_match and sets["permissions"]:
            perm_cond = self._overlap_condition(
                self.config.permissions_field, sets["permissions"]
            )

        if mode == AccessMode.OR:
            should: List[Any] = []
            if role_cond:
                should.append(role_cond)
            if dept_parts:
                should.extend(dept_parts)
            if perm_cond:
                should.append(perm_cond)
            if not should:
                return None
            return rest.Filter(min_should=_make_min_should(should, min_count=1))

        if mode == AccessMode.AND:
            must_parts: List[Any] = []
            if sets["roles"] and sets["departments"]:
                if role_cond:
                    must_parts.append(role_cond)
                if dept_parts:
                    must_parts.extend(dept_parts)
            elif sets["roles"]:
                if role_cond:
                    must_parts.append(role_cond)
            elif sets["departments"]:
                if dept_parts:
                    must_parts.extend(dept_parts)
            if not must_parts:
                return None
            if len(must_parts) == 1:
                return must_parts[0]
            return rest.Filter(must=must_parts)

        if mode == AccessMode.HYBRID:
            if self.config.hybrid_both_mode == HybridBothMode.OR:
                should = []
                if role_cond:
                    should.append(role_cond)
                if dept_parts:
                    should.extend(dept_parts)
                if perm_cond:
                    should.append(perm_cond)
                if not should:
                    return None
                return rest.Filter(min_should=_make_min_should(should, min_count=1))

            # hybrid_both = and
            should = []
            both_must: List[Any] = []
            if role_cond:
                both_must.append(role_cond)
            if dept_parts:
                both_must.extend(dept_parts)
            if len(both_must) >= 2:
                should.append(rest.Filter(must=both_must))
            if role_cond:
                role_only = [
                    role_cond,
                    rest.IsEmptyCondition(
                        is_empty=rest.PayloadField(key=self.config.departments_field)
                    ),
                ]
                should.append(rest.Filter(must=role_only))
            if dept_parts:
                dept_only = list(dept_parts) + [
                    rest.IsEmptyCondition(
                        is_empty=rest.PayloadField(key=self.config.roles_field)
                    )
                ]
                should.append(rest.Filter(must=dept_only))
            if perm_cond:
                should.append(perm_cond)
            if not should:
                return None
            return rest.Filter(min_should=_make_min_should(should, min_count=1))

        return None

    def _overlap_condition(self, field: str, values: Set[str]) -> Optional[Any]:
        if not values or rest is None:
            return None
        return rest.FieldCondition(
            key=field,
            match=rest.MatchAny(any=sorted(values)),
        )

    def _department_conditions(self, departments: Set[str]) -> List[Any]:
        if not departments or rest is None:
            return []
        primary = self._overlap_condition(self.config.departments_field, departments)
        parts: List[Any] = []
        if primary:
            parts.append(primary)
        if self.config.support_legacy_department:
            parts.append(
                rest.FieldCondition(
                    key=self.config.legacy_department_field,
                    match=rest.MatchAny(any=sorted(departments)),
                )
            )
        return parts

    def _status_condition(self) -> Any:
        return rest.FieldCondition(
            key=self.config.status_field,
            match=rest.MatchValue(value=self.config.published_status_value),
        )

    def _forbidden_filter(self, extra_must: List[Any]) -> Any:
        must = list(extra_must)
        must.append(
            rest.FieldCondition(
                key=self.config.roles_field,
                match=rest.MatchValue(value=self.FORBIDDEN_VALUE),
            )
        )
        return rest.Filter(must=must)