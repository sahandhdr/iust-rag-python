# rag_engine.py
"""
RAGEngine — LangGraph retrieve → generate with Hybrid Qdrant + RBAC + session memory.

Memory:
  - Per thread_id = user:{user_id}:session:{session_id}
  - Last N turns injected into the LLM prompt

Retrieve resilience:
  - If embedding/Qdrant fails, return empty context (do not 500 the whole ask)

Query expansion (retrieval only):
  - Institutional synonyms (فاوا ↔ مرکز کامپیوتر)

People-query completeness:
  - Reorder context blocks so presidency/staff sections are not buried
  - Extract name/title lines from full org_context and inject as mandatory checklist
  - Extra HumanMessage instruction for people questions
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from qdrant_client.http import models as rest

from auth.rbac import RBACManager, UserContext
from config.settings import get_settings
from core.hybrid_retrieval import hybrid_search
from models.llm import get_llm

from core.conversation_memory import build_conversation_memory_from_settings

logger = logging.getLogger(__name__)

EMPTY_KNOWLEDGE_ANSWER = (
    "در پایگاه دانش و اسناد منتشرشدهٔ مرکز کامپیوتر دانشگاه علم و صنعت ایران، "
    "اطلاعاتی در این مورد پیدا نشد. لطفاً سؤال را دقیق‌تر بپرسید یا با پشتیبانی مرکز تماس بگیرید."
)

DEFAULT_MAX_HISTORY_TURNS = 12

_RETRIEVAL_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    (
        "فاوا",
        "فناوری اطلاعات",
        "فناوری اطلاعات و ارتباطات",
        "مرکز فاوا",
        "مرکز فناوری اطلاعات",
        "مرکز فناوری اطلاعات و ارتباطات",
        "مرکز کامپیوتر",
        "مرکز رایانه",
        "مرکز کامپیوتر دانشگاه",
    ),
)

_PEOPLE_QUERY_MARKERS = (
    "افراد",
    "فرد",
    "پرسنل",
    "کارکنان",
    "کارمند",
    "نیروها",
    "نیروی",
    "ریاست",
    "رئیس",
    "معاون",
    "مدیران",
    "مدیر",
    "کارشناس",
    "مسئول",
    "مشاور",
    "چه کسانی",
    "کی هست",
    "اسم",
    "نام ",
)

_CONTEXT_PRIORITY_MARKERS = (
    "ریاست",
    "رئیس مرکز",
    "لاریجانی",
    "معاونت",
    "معاون مرکز",
    "مدیران ارشد",
)

# Lines likely to carry person + role
_PERSON_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:[-*•·]|\d+[\.\)]\s*)?.{0,120}?"
    r"(?:دکتر|مهندس|ریاست|رئیس|معاون|مشاور|مسئول|کارشناس|مدیر)."
    r"{2,200}$"
)


def expand_query_for_retrieval(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return q

    q_norm = q
    extras: List[str] = []

    for group in _RETRIEVAL_SYNONYM_GROUPS:
        hit = any(term and term in q_norm for term in group)
        if not hit:
            continue
        for term in group:
            if term and term not in q_norm and term not in extras:
                extras.append(term)

    if not extras:
        return q

    extras = extras[:6]
    return f"{q} (هم‌معنی سازمانی: {' ؛ '.join(extras)})"


def is_people_query(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return any(m in q for m in _PEOPLE_QUERY_MARKERS)


def reorder_org_context_for_people(org_context: str) -> str:
    """
    Move blocks that mention presidency / key org structure to the front
    so the LLM does not anchor only on a short partial staff list ranked higher by RRF.
    """
    text = (org_context or "").strip()
    if not text:
        return text

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        return text

    priority: List[str] = []
    rest: List[str] = []
    for b in blocks:
        if any(m in b for m in _CONTEXT_PRIORITY_MARKERS):
            priority.append(b)
        else:
            rest.append(b)

    if not priority:
        return text

    ordered = priority + rest
    return "\n\n".join(ordered)


def extract_person_lines(org_context: str, *, max_lines: int = 40) -> List[str]:
    """
    Deterministic extraction of lines that look like name/title mentions.
    Used as a mandatory checklist in the system prompt for people questions.
    """
    text = (org_context or "").strip()
    if not text:
        return []

    found: List[str] = []
    seen = set()

    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 8 or len(line) > 240:
            continue
        # Must look role/title related
        if not any(
            k in line
            for k in (
                "دکتر",
                "مهندس",
                "ریاست",
                "رئیس",
                "معاون",
                "مشاور",
                "مسئول",
                "کارشناس",
                "مدیر",
            )
        ):
            continue
        # Prefer substantive lines (not pure headings with no name-like content)
        norm = re.sub(r"\s+", " ", line)
        key = norm.casefold() if hasattr(norm, "casefold") else norm.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(norm)
        if len(found) >= max_lines:
            break

    # Also try regex pass for bullet-like rows missed above
    if len(found) < 3:
        for m in _PERSON_LINE_RE.finditer(text):
            norm = re.sub(r"\s+", " ", m.group(0).strip())
            key = norm.casefold() if hasattr(norm, "casefold") else norm.lower()
            if key not in seen and len(norm) >= 8:
                seen.add(key)
                found.append(norm)
            if len(found) >= max_lines:
                break

    return found


class RAGState(TypedDict):
    question: str
    session_id: str
    user_context: Dict[str, Any]
    user_file_content: Optional[str]
    msg_id: Optional[str]
    org_context: str
    sources: List[Dict[str, Any]]
    answer: str
    thread_id: str
    chat_history_text: str


BASE_RULES = (
    "قوانین مهم و غیرقابل نقض:\n"
    "- فقط و فقط بر اساس اطلاعات موجود در بخش «دانش سازمانی» و (در صورت وجود) "
    "«محتوای فایل کاربر» و «تاریخچه گفتگو» و در صورت وجود «فهرست الزامی افراد استخراج‌شده» پاسخ بده.\n"
    "- هیچ ساعت کاری، نام فرد سازمانی، سمت، شماره تماس، آدرس یا واقعیت سازمانی را از دانش عمومی "
    "یا حدس خودت اضافه نکن.\n"
    "- اگر در دانش سازمانی چیزی مرتبط نیست، برای سؤالات سازمانی بگو اطلاعات در اسناد موجود نیست؛ "
    "جزئیات ساختگی ننویس.\n"
    "- از تاریخچه گفتگو فقط برای پیوستگی مکالمه (مثل نام کاربر یا ترجیحات گفته‌شده در همین نشست) استفاده کن.\n"
    "- اگر در دانش سازمانی اطلاعات متناقض وجود دارد (مثلاً دو ساعت کاری مختلف):\n"
    "  • هر دو نسخه را ذکر کن.\n"
    "  • تا حد امکان منبع هر نسخه را مشخص کن.\n"
    "  • بگو برای تأیید نهایی با مرکز تماس بگیرند.\n"
    "\n"
    "قوانین ویژهٔ فهرست افراد / سمت‌ها / ساختار سازمانی (اجباری):\n"
    "- اگر بخش «فهرست الزامی افراد استخراج‌شده از اسناد» وجود دارد، برای سؤال دربارهٔ افراد "
    "باید همهٔ موارد آن فهرست در پاسخ بیاید (مگر کاملاً بی‌ربط به سؤال).\n"
    "- اگر چند فهرست ناقص در دانش سازمانی هست، اتحاد بگیر؛ کوتاه‌ترین فهرست را به‌عنوان لیست کامل قبول نکن.\n"
    "- ریاست مرکز در صورت وجود در دانش سازمانی یا فهرست الزامی حتماً ذکر شود.\n"
    "- کوتاه‌نویسی به معنای حذف نام یا سمت مجاز نیست.\n"
    "\n"
    "- پاسخ را دوستانه، روان و حرفه‌ای بنویس.\n"
    "- عین جملات اسناد را کپی نکن؛ مضمون را با لحن پشتیبانی دانشگاه بازنویسی کن.\n"
)


class SessionConversationMemory:
    """Process-local conversation buffer (one uvicorn worker)."""

    def __init__(self, max_turns: int = DEFAULT_MAX_HISTORY_TURNS) -> None:
        self._max_turns = max(1, int(max_turns))
        self._lock = threading.Lock()
        self._store: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=self._max_turns)
        )

    def get_turns(self, thread_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._store.get(thread_id, ()))

    def append_turn(self, thread_id: str, human: str, ai: str) -> None:
        human = (human or "").strip()
        ai = (ai or "").strip()
        if not human:
            return
        with self._lock:
            self._store[thread_id].append((human, ai))

    def format_for_prompt(self, thread_id: str) -> str:
        turns = self.get_turns(thread_id)
        if not turns:
            return ""
        lines: List[str] = []
        for human, ai in turns:
            lines.append(f"کاربر: {human}")
            if ai:
                lines.append(f"دستیار: {ai}")
        return "\n".join(lines)

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._store.pop(thread_id, None)


_conversation_memory = build_conversation_memory_from_settings()


class RAGEngine:
    def __init__(self) -> None:
        logger.info("Initializing RAG Engine (Hybrid Qdrant + RBAC + session memory)...")
        self.llm = get_llm()
        self.rbac_manager = RBACManager()
        self.retrieval_k = get_settings().ai.retrieval_k
        self.memory = _conversation_memory
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

    def _has_org_evidence(
        self, org_context: str, sources: List[Dict[str, Any]]
    ) -> bool:
        if sources and len(sources) > 0:
            return True
        if org_context and str(org_context).strip():
            return True
        return False

    def _has_user_file(self, user_file_content: Optional[str]) -> bool:
        return bool(user_file_content and str(user_file_content).strip())

    def _prepare_org_context_for_question(
        self, question: str, org_context: str
    ) -> Tuple[str, List[str]]:
        """
        Returns (possibly reordered org_context, extracted person lines).
        """
        ctx = org_context or ""
        people = is_people_query(question)
        if people:
            ctx = reorder_org_context_for_people(ctx)
        person_lines = extract_person_lines(ctx) if people else []
        if people:
            logger.info(
                "People-query context prep | person_lines=%s | has_presidency=%s",
                len(person_lines),
                any("ریاست" in x or "لاریجانی" in x for x in person_lines),
            )
        return ctx, person_lines

    async def _retrieve_for_query(
        self,
        question: str,
        user_context: UserContext,
        session_id: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        retrieval_query = expand_query_for_retrieval(question)
        logger.info(
            "Hybrid retrieving org context | session=%s | retrieval_query_len=%s",
            session_id,
            len(retrieval_query),
        )

        def _run():
            return hybrid_search(
                retrieval_query,
                user_context,
                k=self.retrieval_k,
            )

        try:
            org_context, sources = await asyncio.to_thread(_run)
            return org_context or "", sources or []
        except Exception as exc:
            logger.exception(
                "Retrieval failed (embedding/Qdrant) | session=%s | err=%s",
                session_id,
                exc,
            )
            return "", []

    def _build_system_prompt(
        self,
        org_context: str,
        user_file_content: Optional[str],
        chat_history_text: str,
        *,
        dialogue_only: bool = False,
        person_lines: Optional[List[str]] = None,
    ) -> str:
        history_block = ""
        if chat_history_text and chat_history_text.strip():
            history_block = (
                f"\n\nتاریخچه گفتگوی همین نشست:\n{chat_history_text.strip()}\n"
            )

        personnel_block = ""
        if person_lines:
            bullets = "\n".join(f"- {line}" for line in person_lines)
            personnel_block = (
                "\n\n### فهرست الزامی افراد استخراج‌شده از اسناد\n"
                "(برای سؤال دربارهٔ افراد / سمت‌ها باید همهٔ موارد مرتبط زیر در پاسخ بیایند؛ "
                "هیچ‌کدام را به‌خاطر کوتاه‌نویسی حذف نکن. اگر ریاست در فهرست است حتماً ذکر شود.)\n"
                f"{bullets}\n"
            )

        if dialogue_only:
            return (
                "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
                "در این نوبت دانش سازمانی بازیابی‌شده خالی است یا در دسترس نیست.\n"
                f"{BASE_RULES}\n"
                "- برای سؤالات سازمانی بگو در اسناد منتشرشده چیزی پیدا نشد.\n"
                "- برای پیوستگی مکالمه (نام کاربر، موارد گفته‌شده در همین چت) از تاریخچه استفاده کن.\n"
                f"{history_block}"
                "دانش سازمانی:\n(خالی)\n"
            )

        if user_file_content:
            return (
                "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
                "دو منبع اطلاعاتی در اختیار داری:\n"
                "۱) دانش سازمانی رسمی (اسناد مرکز)\n"
                "۲) محتوای فایلی که کاربر آپلود کرده\n\n"
                f"{BASE_RULES}\n"
                "- در سؤالات مربوط به دانشگاه و مرکز، اولویت با دانش سازمانی رسمی است.\n"
                "- اگر فایل کاربر با اسناد رسمی تعارض دارد، اسناد رسمی را مقدم بدان و تعارض را ذکر کن.\n"
                "- اگر سؤال مستقیماً دربارهٔ فایل آپلود‌شده است، آن را با دقت تحلیل کن.\n"
                "- اگر دانش سازمانی خالی است و فقط فایل کاربر موجود است، فقط بر اساس فایل پاسخ بده "
                "و ادعا نکن که از اسناد رسمی مرکز آمده است.\n"
                f"{history_block}"
                f"{personnel_block}"
                f"دانش سازمانی:\n{org_context}\n\n"
                f"محتوای فایل کاربر:\n{user_file_content}"
            )

        return (
            "تو یک دستیار پشتیبانی هوشمند مرکز کامپیوتر دانشگاه علم و صنعت ایران هستی.\n\n"
            "فقط به دانش سازمانی رسمی (اسناد مرکز) دسترسی داری.\n\n"
            f"{BASE_RULES}\n"
            f"{history_block}"
            f"{personnel_block}"
            f"دانش سازمانی:\n{org_context}"
        )

    def _human_message_for_question(self, question: str) -> str:
        q = (question or "").strip()
        if not is_people_query(q):
            return q
        return (
            f"{q}\n\n"
            "[دستور سیستم برای این نوبت] "
            "اگر در دانش سازمانی یا فهرست الزامی افراد، ریاست مرکز یا نام‌هایی مثل "
            "لاریجانی و سایر دکتر/مهندس‌ها آمده، همه را در پاسخ افراد بیاور. "
            "به یک فهرست ناقص ابتدای متن اکتفا نکن."
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
        org_context_raw = state.get("org_context", "") or ""
        sources = state.get("sources") or []
        history_text = state.get("chat_history_text", "") or ""
        thread_id = state.get("thread_id") or ""

        org_context, person_lines = self._prepare_org_context_for_question(
            question, org_context_raw
        )

        has_org = self._has_org_evidence(org_context, sources)
        has_file = self._has_user_file(user_file)
        dialogue_only = not has_org and not has_file

        system_text = self._build_system_prompt(
            org_context,
            user_file if has_file else None,
            history_text,
            dialogue_only=dialogue_only,
            person_lines=person_lines if not dialogue_only else None,
        )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=self._human_message_for_question(question)),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            answer = getattr(response, "content", None) or str(response)
            if isinstance(answer, list):
                answer = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in answer
                )
            answer = str(answer).strip() or EMPTY_KNOWLEDGE_ANSWER
        except Exception:
            logger.exception("LLM invoke failed | session=%s", state["session_id"])
            answer = EMPTY_KNOWLEDGE_ANSWER

        if thread_id:
            self.memory.append_turn(thread_id, question, answer)

        return {"answer": answer}

    async def query(
        self,
        question: str,
        session_id: str,
        user_context: UserContext,
        user_file_content: Optional[str] = None,
        msg_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        user_context_dict = self._user_context_to_dict(user_context)
        thread_id = self._build_thread_id(user_context, session_id)
        config = {"configurable": {"thread_id": thread_id}}
        history_text = self.memory.format_for_prompt(thread_id)

        initial_state: RAGState = {
            "question": question,
            "session_id": session_id,
            "user_context": user_context_dict,
            "user_file_content": user_file_content,
            "msg_id": msg_id,
            "org_context": "",
            "sources": [],
            "answer": "",
            "thread_id": thread_id,
            "chat_history_text": history_text,
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
        thread_id = self._build_thread_id(user_context, session_id)
        history_text = self.memory.format_for_prompt(thread_id)

        org_context_raw, sources = await self._retrieve_for_query(
            question=question,
            user_context=user_context,
            session_id=session_id,
        )
        yield ("sources", sources)

        org_context, person_lines = self._prepare_org_context_for_question(
            question, org_context_raw
        )

        has_org = self._has_org_evidence(org_context, sources)
        has_file = self._has_user_file(user_file_content)
        dialogue_only = not has_org and not has_file

        system_text = self._build_system_prompt(
            org_context,
            user_file_content if has_file else None,
            history_text,
            dialogue_only=dialogue_only,
            person_lines=person_lines if not dialogue_only else None,
        )
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=self._human_message_for_question(question)),
        ]

        full_parts: List[str] = []
        try:
            async for chunk in self.llm.astream(messages):
                text = getattr(chunk, "content", None)
                if text is None:
                    continue
                if isinstance(text, list):
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in text
                    )
                if not text:
                    continue
                full_parts.append(str(text))
                yield ("token", text)
        except Exception:
            logger.exception("LLM stream failed | session=%s", session_id)
            full_parts = [EMPTY_KNOWLEDGE_ANSWER]
            yield ("token", EMPTY_KNOWLEDGE_ANSWER)

        full_answer = "".join(full_parts).strip() or EMPTY_KNOWLEDGE_ANSWER
        self.memory.append_turn(thread_id, question, full_answer)
        yield ("done", {"answer": full_answer})


rag_engine_instance = RAGEngine()


def get_rag_engine() -> RAGEngine:
    return rag_engine_instance