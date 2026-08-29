# tests/test_rag.py
import asyncio
import logging
import os
import tempfile
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def test_config():
    print("1. تست Config...")
    print(f"   LLM: {settings.ai.llm_provider} | Model: {settings.ai.llm_model}")
    print(f"   Embedding: {settings.ai.embedding_provider} | Model: {settings.ai.embedding_model}")
    print(f"   Qdrant: {settings.db.qdrant_collection}")
    assert settings.ai.llm_provider, "LLM Provider تنظیم نشده"
    print("   ✅ Config OK")


async def test_vector_store():
    print("2. تست Vector Store...")
    from create_database import get_vector_store
    vector_store = get_vector_store()
    assert vector_store is not None
    print("   ✅ Vector Store OK")


async def test_ingestion():
    print("3. تست Ingestion...")
    from ingest_documents import document_ingestor

    test_content = "# تست سند\n\nاین یک سند آزمایشی برای دانشگاه علم و صنعت است."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(test_content)
        temp_path = f.name

    try:
        success = document_ingestor.process_single_file(
            file_path=temp_path,
            department="public",
            doc_uuid="test-uuid-integration"
        )
        assert success, "Ingestion ناموفق بود"
        print("   ✅ Ingestion OK")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def test_rbac_sync():
    print("4. تست RBAC & QdrantSync...")
    from auth_rbac import qdrant_sync
    results = qdrant_sync.search_by_doc_uuid("test-uuid-integration", limit=5)
    print(f"   ✅ Sync Search OK - تعداد چانک: {len(results)}")


async def test_llm():
    print("5. تست LLM...")
    from get_llm import get_llm
    llm = get_llm(temperature=0.3)
    response = llm.invoke("فقط بگو 'تست LLM موفق است'")
    assert "تست LLM موفق" in response.content
    print("   ✅ LLM OK")


async def run_full_test():
    print("🚀 شروع تست کامل سیستم RAG - Phase 1\n")
    try:
        await test_config()
        await test_vector_store()
        await test_ingestion()
        await test_rbac_sync()
        await test_llm()
        print("\n🎉 تمام تست‌ها با موفقیت انجام شد! آماده اتصال به Laravel.")
    except Exception as e:
        logger.exception("تست ناموفق")
        print(f"❌ خطا در تست: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_full_test())