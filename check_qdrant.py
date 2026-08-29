# check_qdrant.py
import sys
import os
import logging
from typing import Optional

# اضافه کردن مسیر پروژه برای import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_database import get_vector_store
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)


def check_document_in_qdrant(doc_uuid: str) -> Optional[dict]:
    """
    بررسی وجود سند در Qdrant بر اساس doc_uuid.
    ابزار کاربردی برای debug و تست (Phase 1).
    """
    if not doc_uuid or not isinstance(doc_uuid, str):
        print("❌ doc_uuid معتبر نیست.")
        return None

    try:
        vector_store = get_vector_store()

        # فیلتر دقیق بر اساس doc_uuid
        filter_condition = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.doc_uuid",
                    match=rest.MatchValue(value=doc_uuid.strip())
                )
            ]
        )

        results = vector_store.similarity_search(
            query="test",  # query dummy برای جستجو
            k=20,
            filter=filter_condition
        )

        if results:
            print(f"✅ سند پیدا شد - doc_uuid: {doc_uuid}")
            print(f"تعداد chunkها: {len(results)}")
            for i, doc in enumerate(results):
                print(f"  Chunk {i + 1}: source={doc.metadata.get('source')}, "
                      f"department={doc.metadata.get('department')}, "
                      f"chunk_index={doc.metadata.get('chunk_index')}")
            return {
                "found": True,
                "doc_uuid": doc_uuid,
                "chunk_count": len(results),
                "samples": [doc.metadata for doc in results[:3]]
            }
        else:
            print(f"❌ سند پیدا نشد - doc_uuid: {doc_uuid}")
            return {"found": False, "doc_uuid": doc_uuid}

    except Exception as e:
        logger.exception(f"خطا در چک کردن سند {doc_uuid}")
        print(f"❌ خطای فنی: {e}")
        return None


def list_all_documents(limit: int = 10):
    """لیست کردن تعدادی سند موجود در Qdrant (برای debug)."""
    try:
        vector_store = get_vector_store()
        # جستجوی ساده برای گرفتن نمونه‌ها
        results = vector_store.similarity_search(query=" ", k=limit)

        print(f"\n📋 {len(results)} سند/چانک نمونه:")
        seen_docs = set()
        for doc in results:
            duuid = doc.metadata.get("doc_uuid")
            if duuid and duuid not in seen_docs:
                seen_docs.add(duuid)
                print(f"   • doc_uuid: {duuid} | department: {doc.metadata.get('department')} | "
                      f"source: {doc.metadata.get('source')}")
    except Exception as e:
        print(f"خطا در لیست اسناد: {e}")


if __name__ == "__main__":
    print("=== Qdrant Document Checker (doc_uuid) ===\n")

    action = input("عملیات (1=چک سند خاص | 2=لیست اسناد): ").strip()

    if action == "1":
        doc_uuid = input("doc_uuid سند را وارد کنید: ").strip()
        if doc_uuid:
            check_document_in_qdrant(doc_uuid)
    elif action == "2":
        limit = int(input("تعداد اسناد برای نمایش (پیش‌فرض 10): ") or "10")
        list_all_documents(limit)
    else:
        print("عملیات نامعتبر.")