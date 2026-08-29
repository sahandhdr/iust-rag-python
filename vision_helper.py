# vision_helper.py
import base64
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

AnalysisMode = Literal["ocr", "vision"]


class VisionHelper:
    def __init__(self):
        if settings.ai.vision_provider.lower() != "gapgpt":
            logger.warning(f"Vision provider is '{settings.ai.vision_provider}', but expected 'gapgpt'.")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_image(self, image_path: str, analysis_mode: AnalysisMode = "ocr") -> str:
        """تحلیل تصویر با تمرکز ویژه روی متن فارسی."""
        if analysis_mode == "ocr":
            return self._ocr_with_vision_model(image_path)
        return self._vision_analyze(image_path)

    def _ocr_with_vision_model(self, image_path: str) -> str:
        """OCR بهبودیافته برای فارسی (RTL + ساختار حفظ شود)."""
        logger.info(f"Starting Persian OCR on: {image_path}")

        model = ChatOpenAI(
            model=settings.ai.gapgpt_vision_model,
            api_key=settings.ai.gapgpt_api_key,
            base_url=settings.ai.gapgpt_base_url,
            temperature=0.0
        )

        base64_image = self._encode_image(image_path)

        prompt = """شما یک OCR حرفه‌ای برای زبان فارسی هستید.
متن کامل داخل تصویر را دقیقاً استخراج کنید.
- جهت متن راست به چپ (RTL) را حفظ کنید.
- جدول‌ها، لیست‌ها و ساختار را تا حد ممکن نگه دارید.
- فقط متن خام برگردانید، هیچ توضیح اضافی ندهید."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        try:
            response = model.invoke([SystemMessage(content="You are an accurate Persian OCR engine."), message])
            return response.content.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"[OCR Error]: {e}"

    def _vision_analyze(self, image_path: str) -> str:
        """تحلیل عمومی تصویر."""
        # مشابه قبل...
        pass  # فعلاً فقط OCR مهم است


vision_helper_instance = VisionHelper()