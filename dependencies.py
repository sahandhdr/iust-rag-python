# dependencies.py
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth_rbac import LaravelAuthenticator, UserContext

# تعریف سیستم Security توکن در FastAPI
oauth2_scheme = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> UserContext:
    """
    دریافت توکن از هدرها و تایید آن از طریق متد اختصاصی اتصال به لاراول.
    این تابع مستقیماً به عنوان Dependency در روترها استفاده می‌شود.
    """
    token = credentials.credentials

    # ارسال به احراز هویت لاراول (LaravelAuthenticator) که در auth_rbac.py نوشتیم
    # این متد در صورت خطا به طور خودکار HTTPException برمی‌گرداند و در صورت موفقیت UserContext را می‌دهد
    return await LaravelAuthenticator.verify_token(token)
