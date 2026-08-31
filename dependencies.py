# dependencies.py
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth_rbac import LaravelAuthenticator, UserContext
from config import get_settings

oauth2_scheme = HTTPBearer(auto_error=False)


def _internal_admin_context() -> UserContext:
    """
    کاربر سیستمی برای فراخوانی Laravel → Python.
    بدون callback به verify-token (جلوگیری از deadlock روی artisan serve).
    """
    return UserContext(
        user_id=1,
        username="laravel_internal",
        roles=["admin", "developer"],
        departments=[],
        permissions=["all", "documents.read.all", "rbac.bypass"],
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme),
    x_internal_key: str | None = Header(default=None, alias="X-Internal-Key"),
) -> UserContext:
    """
    اولویت:
      1) X-Internal-Key مطابق LARAVEL__INTERNAL_API_KEY → UserContext ادمین سیستمی
      2) Authorization: Bearer → LaravelAuthenticator.verify_token
    """
    settings = get_settings()
    expected = (settings.laravel.internal_api_key or "").strip()

    if expected and x_internal_key and x_internal_key.strip() == expected:
        return _internal_admin_context()

    if credentials is None or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="توکن احراز هویت ارسال نشده است.",
        )

    return await LaravelAuthenticator.verify_token(credentials.credentials)