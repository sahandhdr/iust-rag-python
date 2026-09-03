"""
RAGEngine — LangGraph retrieve → generate with RBAC-filtered Qdrant search.

Security:
  - UserContext reconstructed fail-closed from checkpoint dict
  - Retrieval always goes through RBACManager.build_qdrant_filter
    (DocumentAccessPolicy: or / and / hybrid)
  - thread_id scoped to user_id + session_id

Phase-1:
  - retrieval_k from config
  - msg_id must NOT filter public knowledge retrieve
  - query()      → one-shot answer
  - query_stream() → async generator for SSE (sources, token, done)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from qdrant_client.http import models as rest

from auth_rbac import RBACManager, UserContext
from config import get_settings
from create_database import get_vector_store
from get_llm import get_llm

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """Checkpoint-safe graph state (user_context as dict, not object)."""

    question: str
    session_id: str
    user_context: Dict[str, Any]
    user_file_content: Optional[str]
    msg_id: Optional[str]
    org_context: str
    sources: List[Dict[str, Any]]
    answer: str


BASE_RULES = (
    "قوانین مهم:\n"
    "- فقط بر اساس اطلاعات موجود در context پاسخ بده. هیچ واقعیتی را از خودت نساز.\n"
    "- اگر اطلاعات کافی نیست، صریح بگو که در اسناد موجود پاسخی پیدا نکردی.\n"
    "- اگر در context اطلاعات متناقض یا متفاوت وجود دارد (مثلاً ساعات کاری متفاوت):\n"
    "  • حتماً هر دو نسخه را ذکر کن.\n"
    "  • منبع مربوط به هر نسخه را تا حد امکان مشخص کن.\n"
    "  • بگو که هر دو می‌توانند معتبر باشند یا نیاز به بررسی بیشتر دارند.\n"
    "- اگر سؤال دربارهٔ افراد، سمت‌ها، مدیران یا ساختار سازمانی است و در context نام افراد آمده:\n"
    "  • همهٔ نام‌ها و سمت‌های مرتبط موجود در context را ذکر کن.\n"
    "  • هیچ نامی را به بهانهٔ خلاصه‌نویسی حذف نکن.\n"
    "  • اگر ریاست/معاونت/کارشناس در context هست، همگی باید در پاسخ باشند.\n"
    "- پاسخ را به زبان دوستانه، روان و حرفه‌ای بنویس. می‌توانی مختصر باشی، "
    "اما مختصر بودن هرگز به معنای حذف نام‌ها یا حقایق موجود در context نیست.\n"
    "- عین جملات اسناد را کپی نکن؛ مضمون را با لحن پشتیبانی دانشگاه بازنویسی کن.\n"
    "- از لحن رسمی اما صمیمی استفاده کن (مناسب پشتیبانی دانشگاه).\n"
)


class RAGEngine:
    def __init__(self) -> None:
        logger.info("Initializing RAG Engine (Qdrant + RBAC Policy)...")
        self.llm = get_llm()
        self.db = get_vector_store()
        self.rbac_manager = RBACManager()
        self.retrieval_k = get_settings().ai.retrieval_k
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=MemorySaver())

    def _user_context_to_dict(self, user_context: UserContext) -> Dict[str, Any]:
        if hasattr(user_context, "model_dump"):
            data = user_context.model_dump()
        elif hasattr(user_context, "dict"):
            data = user_context.dict()
        else:
            raise ValueError("Unsupported UserContext type for serialization.")
        if not isinstance(data, dict):
            raise ValueError("Serialized user_context must be a dict.")
        return data

    def _dict_to_user_context(self, data: Dict[str, Any]) -> UserContext:
        if not isinstance(data, dict):
            raise ValueError("user_context in state must be a dict.")
        try:
            if hasattr(UserContext, "model_validate"):
                return UserContext.model_validate(data)
            return UserContext(**data)
        except Exception as exc:
            logger.exception("Failed to reconstruct UserContext from state.")
            raise ValueError("Invalid user_context in checkpoint/state.") from exc

    def _build_thread_id(self, user_context: UserContext, session_id: str) -> str:
        user_id = getattr(user_context, "user_id", None)
        if not user_id:
            raise ValueError("UserContext must contain a valid user_id.")
        if not session_id:
            raise ValueError("session_id is required.")
        return f"user:{user_id}:session:{session_id}"

    def _merge_msg_filter(
        self,
        qdrant_filter: Optional[rest.Filter],
        msg_id: str,
    ) -> rest.Filter:
        """
        Legacy helper — DO NOT use for public knowledge ask.
        Kept for possible future message-scoped file chunks only.
        """
        msg_condition = rest.FieldCondition(
            key="metadata.msg_id",
            match=rest.MatchValue(value=msg_id),
        )
        if qdrant_filter is None:
            return rest.Filter(must=[msg_condition])

        must = list(qdrant_filter.must or [])
        must.append(msg_condition)
        return rest.Filter(
            must=must or None,
            should=list(qdrant_filter.should) if qdrant_filter.should else None,
            must_not=list(qdrant_filter.must_not) if qdrant_filter.must_not else None,
            min_should=getattr(qdrant_filter, "min_should", None),
        )

    def _sources_from_results(self, results: list) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for doc, score in results:
            sources.append(
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "doc_uuid": doc.metadata.get("doc_uuid"),
                    "roles": doc.metadata.get("roles"),
                    "departments": doc.metadata.get("departments"),
                    "status": doc.metadata.get("status"),
                    "relevance_score": float(score),
                }
            )
        return sources

    async def _retrieve_for_query(
        self,
        question: str,
        user_context: UserContext,
        session_id: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Shared retrieve for graph node and stream path. No msg_id filter."""
        logger.info("Retrieving org context | session=%s", session_id)

        qdrant_filter = self.rbac_manager.build_qdrant_filter(user_context)

        logger.debug(
            "Qdrant filter applied | user_id=%s | roles=%s | depts=%s | filter=%s",
            user_context.user_id,
            sorted(user_context.roles),
            sorted(user_context.departments),
            qdrant_filter,
        )

        results = await self.db.asimilarity_search_with_score(
            question,
            k=self.retrieval_k,
            filter=qdrant_filter,
        )

        org_context = "\n\n".join(doc.page_content for doc, _ in results)
        sources = self._sources_from_results(results)
        return org_context, sources

    def _build_system_prompt(
        self,
        org_context: str,
        user_file_content: Optional[str],
    ) -> str:
        if user_file_content:
            return (
                "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
                "دو منبع اطلاعاتی در اختیار داری:\n"
                "۱) دانش سازمانی رسمی (اسناد مرکز)\n"
                "۲) محتوای فایلی که کاربر آپلود کرده\n\n"
                f"{BASE_RULES}\n"
                "- در سؤالات مربوط به دانشگاه و مرکز کامپیوتر، اولویت با دانش سازمانی رسمی است.\n"
                "- اگر فایل کاربر با اسناد رسمی تعارض دارد، اسناد رسمی را مقدم بدان و تعارض را ذکر کن.\n"
                "- اگر سؤال مستقیماً در مورد فایل آپلود‌شده است، آن را با دقت تحلیل کن.\n\n"
                f"دانش سازمانی:\n{org_context}\n\n"
                f"محتوای فایل کاربر:\n{user_file_content}"
            )
        return (
            "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
            "فقط به دانش سازمانی رسمی (اسناد مرکز) دسترسی داری.\n\n"
            f"{BASE_RULES}\n"
            f"دانش سازمانی:\n{org_context}"
        )

    async def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        user_context = self._dict_to_user_context(state["user_context"])
        org_context, sources = await self._retrieve_for_query(
            question=state["question"],
            user_context=user_context,
            session_id=state["session_id"],
        )
        return {"org_context": org_context, "sources": sources}

    async def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        logger.info("Generating answer | session=%s", state["session_id"])

        user_file = state.get("user_file_content")
        question = state["question"]
        org_context = state.get("org_context", "")

        system_prompt = self._build_system_prompt(org_context, user_file)

        if user_file:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{question}")]
            )
            # system already has contexts embedded; template vars only question
            # Re-build with placeholders for chain invoke compatibility:
            system_with_vars = (
                "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
                "دو منبع اطلاعاتی در اختیار داری:\n"
                "۱) دانش سازمانی رسمی (اسناد مرکز)\n"
                "۲) محتوای فایلی که کاربر آپلود کرده\n\n"
                f"{BASE_RULES}\n"
                "- در سؤالات مربوط به دانشگاه و مرکز کامپیوتر، اولویت با دانش سازمانی رسمی است.\n"
                "- اگر فایل کاربر با اسناد رسمی تعارض دارد، اسناد رسمی را مقدم بدان و تعارض را ذکر کن.\n"
                "- اگر سؤال مستقیماً در مورد فایل آپلود‌شده است، آن را با دقت تحلیل کن.\n\n"
                "دانش سازمانی:\n{org_context}\n\n"
                "محتوای فایل کاربر:\n{user_file_content}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_with_vars), ("human", "{question}")]
            )
            chain = prompt | self.llm | StrOutputParser()
            answer = await chain.ainvoke(
                {
                    "org_context": org_context,
                    "user_file_content": user_file,
                    "question": question,
                }
            )
        else:
            system_with_vars = (
                "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
                "فقط به دانش سازمانی رسمی (اسناد مرکز) دسترسی داری.\n\n"
                f"{BASE_RULES}\n"
                "دانش سازمانی:\n{org_context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_with_vars), ("human", "{question}")]
            )
            chain = prompt | self.llm | StrOutputParser()
            answer = await chain.ainvoke(
                {
                    "org_context": org_context,
                    "question": question,
                }
            )

        return {"answer": answer}

    async def query(
        self,
        question: str,
        session_id: str,
        user_context: UserContext,
        user_file_content: Optional[str] = None,
        msg_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """One-shot RAG (non-streaming). msg_id ignored for retrieve."""
        user_context_dict = self._user_context_to_dict(user_context)
        thread_id = self._build_thread_id(user_context, session_id)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: RAGState = {
            "question": question,
            "session_id": session_id,
            "user_context": user_context_dict,
            "user_file_content": user_file_content,
            "msg_id": msg_id,
            "org_context": "",
            "sources": [],
            "answer": "",
        }

        result = await self.app.ainvoke(initial_state, config=config)
        return result.get("answer", ""), result.get("sources", [])

    async def query_stream(
        self,
        question: str,
        session_id: str,
        user_context: UserContext,
        user_file_content: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Streaming RAG for SSE.

        Yields:
          ("sources", list[dict])
          ("token", str)          — may yield many times
          ("done", {"answer": str})
        """
        org_context, sources = await self._retrieve_for_query(
            question=question,
            user_context=user_context,
            session_id=session_id,
        )
        yield ("sources", sources)

        system_text = self._build_system_prompt(org_context, user_file_content)
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=question),
        ]

        full_parts: List[str] = []
        async for chunk in self.llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                continue
            if isinstance(text, list):
                # some providers return content blocks
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if not text:
                continue
            full_parts.append(text)
            yield ("token", text)

        yield ("done", {"answer": "".join(full_parts)})


rag_engine_instance = RAGEngine()


def get_rag_engine() -> RAGEngine:
    return rag_engine_instance