# ingest_documents.py
"""
Document ingestion pipeline for IUST RAG.

Flow:
  extract text → save markdown under data/{department}/ → chunk + upsert Qdrant

Laravel publish/sync must pass:
  doc_uuid, roles, departments, status
so ACL metadata on Qdrant matches MySQL.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pdfplumber
from docling.document_converter import DocumentConverter

from config import get_settings
from vision_helper import vision_helper_instance

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    "pdf", "docx", "pptx", "xlsx", "txt", "md", "csv",
    "jpg", "jpeg", "png", "gif", "bmp", "tiff",
    "html", "xml", "json",
}


class DocumentIngestor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.data_dir = self.settings.ingestion.data_dir
        self.vision = vision_helper_instance

    def _ensure_department_dir(self, department: str) -> str:
        dept = (department or "public").strip().lower()
        dept_path = os.path.join(self.data_dir, dept)
        os.makedirs(dept_path, exist_ok=True)
        return dept_path

    def _ocr_pdf(self, pdf_path: str) -> str:
        """PDF text extract with Vision OCR fallback for scanned pages."""
        md_content: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    md_content.append(f"## Page {page_num}\n\n{text}")
                    continue

                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                        temp_path = temp.name
                    page.to_image(resolution=250).save(temp_path, format="PNG")
                    extracted = self.vision.analyze_image(temp_path, analysis_mode="ocr")
                    if extracted and not str(extracted).startswith("[OCR Error]"):
                        md_content.append(f"## Page {page_num}\n\n{extracted}")
                    else:
                        md_content.append(f"## Page {page_num}\n\n[متن قابل استخراج نبود]")
                except Exception as exc:
                    logger.warning("Vision OCR failed on page %s: %s", page_num, exc)
                    md_content.append(f"## Page {page_num}\n\n[خطا در استخراج]")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass
        return "\n\n".join(md_content)

    def _ocr_image(self, image_path: str) -> str:
        return self.vision.analyze_image(image_path, analysis_mode="ocr")

    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """Extract text for supported formats. Returns None on hard failure."""
        ext = Path(file_path).suffix.lower().lstrip(".")

        try:
            if ext == "pdf":
                return self._ocr_pdf(file_path)
            if ext in {"jpg", "jpeg", "png", "gif", "bmp", "tiff"}:
                return self._ocr_image(file_path)
            if ext in {"docx", "pptx"}:
                converter = DocumentConverter()
                result = converter.convert(file_path)
                return result.document.export_to_markdown()
            if ext in {"txt", "md", "html", "xml", "json"}:
                with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            if ext in {"xlsx", "csv"}:
                logger.warning("Spreadsheet support pending for: %s", file_path)
                return "[محتوای جدول در آینده پشتیبانی خواهد شد]"
            logger.warning("Unsupported extension: %s for file %s", ext, file_path)
            return None
        except Exception as exc:
            logger.error("Text extraction failed for %s: %s", file_path, exc, exc_info=True)
            return None

    def process_single_file(
        self,
        file_path: str,
        department: str = "public",
        doc_uuid: Optional[str] = None,
        roles: Optional[Sequence[str]] = None,
        departments: Optional[Sequence[str]] = None,
        permissions: Optional[Sequence[str]] = None,
        status: str = "published",
        version: int = 1,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Full ingest pipeline.

        Returns:
          {"success": True, ...process_single_document fields}
          {"success": False, "error": "..."}
        """
        try:
            logger.info(
                "Starting ingestion: path=%s | department=%s | doc_uuid=%s | roles=%s | departments=%s | status=%s",
                file_path,
                department,
                doc_uuid or "auto",
                list(roles) if roles else None,
                list(departments) if departments else None,
                status,
            )

            text = self.extract_text_from_file(file_path)
            if not text or not str(text).strip():
                logger.warning("No usable text extracted from %s", file_path)
                return {"success": False, "error": "no_text_extracted"}

            dept_key = (department or "public").strip().lower() or "public"
            dept_dir = self._ensure_department_dir(dept_key)

            stem = Path(file_path).stem
            if doc_uuid and str(doc_uuid).strip():
                safe_uuid = re.sub(r"[^a-zA-Z0-9\-_]", "_", str(doc_uuid).strip())
                md_filename = f"{safe_uuid}.md"
            else:
                md_filename = f"{stem}.md"

            md_path = os.path.join(dept_dir, md_filename)
            with open(md_path, "w", encoding="utf-8") as handle:
                handle.write(text)

            logger.info("Markdown saved: %s", md_path)

            if departments is not None:
                dept_list = [str(d).strip() for d in departments if str(d).strip()]
            else:
                dept_list = [dept_key]

            roles_list = None
            if roles is not None:
                roles_list = [str(r).strip() for r in roles if str(r).strip()]

            perms_list = None
            if permissions is not None:
                perms_list = [str(p).strip() for p in permissions if str(p).strip()]

            from create_database import process_single_document

            result = process_single_document(
                md_path=md_path,
                department=dept_key,
                doc_uuid=doc_uuid,
                roles=roles_list,
                departments=dept_list,
                permissions=perms_list,
                status=status or "published",
                version=version if isinstance(version, int) and version >= 1 else 1,
                overwrite=overwrite,
            )

            logger.info("Document processed successfully: %s", result)
            out: Dict[str, Any] = {"success": True}
            if isinstance(result, dict):
                out.update(result)
            return out

        except Exception as exc:
            logger.error(
                "Failed to process file %s (doc_uuid=%s): %s",
                file_path,
                doc_uuid,
                exc,
                exc_info=True,
            )
            return {"success": False, "error": str(exc)}


document_ingestor = DocumentIngestor()