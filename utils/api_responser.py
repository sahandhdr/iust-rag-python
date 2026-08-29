# api_responser.py
from typing import Any, Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[Any] = None

class ApiResponser:
    @staticmethod
    def success_response(data: Any = None, code: int = 200, message: Optional[str] = "Success") -> JSONResponse:
        # تبدیل set به list برای JSON
        if isinstance(data, dict):
            data = {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
        content = StandardResponse(
            success=True,
            message=message,
            data=data
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=code, content=content)

    @staticmethod
    def error_response(message: str, status_code: int = 400, errors: Any = None) -> JSONResponse:
        content = StandardResponse(
            success=False,
            message=message,
            errors=errors
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=status_code, content=content)