# conversation_memory.py
"""
Persistent session conversation memory for RAG.

Primary store: Redis (survives uvicorn restart, shared across workers).
Fallback: process-local deque if Redis is unavailable.

Key schema:
  iust:rag:mem:{user_id}:{session_id}  -> JSON {turns, summary, updated_at}
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 12
DEFAULT_TTL_SECONDS = 604_800  # 7 days
KEY_PREFIX = "iust:rag:mem"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionConversationMemory:
    """
    Load/save chat turns for a thread_id (user:{id}:session:{sid}).
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://127.0.0.1:6379/0",
        max_turns: int = DEFAULT_MAX_TURNS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_turns = max(2, int(max_turns))
        self._ttl = max(60, int(ttl_seconds))
        self._redis_url = (redis_url or "").strip()
        self._redis = None
        self._redis_ok = False
        self._lock = threading.Lock()
        self._local: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=self._max_turns)
        )
        self._local_summary: Dict[str, str] = {}
        self._connect_redis()

    def _connect_redis(self) -> None:
        if not self._redis_url:
            logger.warning("REDIS__URL empty — conversation memory uses process RAM only")
            return
        try:
            import redis

            client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self._redis = client
            self._redis_ok = True
            logger.info("Conversation memory connected to Redis")
        except Exception as exc:
            self._redis = None
            self._redis_ok = False
            logger.warning(
                "Redis unavailable for conversation memory (%s) — using process RAM fallback",
                exc,
            )

    def _redis_key(self, thread_id: str) -> str:
        # thread_id already: user:{id}:session:{sid}
        safe = thread_id.replace(" ", "_")
        return f"{KEY_PREFIX}:{safe}"

    def _load_payload(self, thread_id: str) -> Dict[str, Any]:
        if self._redis_ok and self._redis is not None:
            try:
                raw = self._redis.get(self._redis_key(thread_id))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:
                logger.warning("Redis get failed for %s: %s", thread_id, exc)
                self._redis_ok = False
        with self._lock:
            turns = list(self._local.get(thread_id, ()))
            summary = self._local_summary.get(thread_id, "")
        return {
            "turns": [{"human": h, "ai": a} for h, a in turns],
            "summary": summary,
            "updated_at": _utcnow_iso(),
        }

    def _save_payload(self, thread_id: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at"] = _utcnow_iso()
        turns = payload.get("turns") or []
        if len(turns) > self._max_turns:
            payload["turns"] = turns[-self._max_turns :]

        if self._redis_ok and self._redis is not None:
            try:
                self._redis.set(
                    self._redis_key(thread_id),
                    json.dumps(payload, ensure_ascii=False),
                    ex=self._ttl,
                )
            except Exception as exc:
                logger.warning("Redis set failed for %s: %s", thread_id, exc)
                self._redis_ok = False

        with self._lock:
            dq: Deque[Tuple[str, str]] = deque(maxlen=self._max_turns)
            for item in payload.get("turns") or []:
                if isinstance(item, dict):
                    dq.append(
                        (
                            str(item.get("human") or "").strip(),
                            str(item.get("ai") or "").strip(),
                        )
                    )
            self._local[thread_id] = dq
            self._local_summary[thread_id] = str(payload.get("summary") or "")

    def get_turns(self, thread_id: str) -> List[Tuple[str, str]]:
        data = self._load_payload(thread_id)
        out: List[Tuple[str, str]] = []
        for item in data.get("turns") or []:
            if isinstance(item, dict):
                h = str(item.get("human") or "").strip()
                a = str(item.get("ai") or "").strip()
                if h:
                    out.append((h, a))
        return out

    def get_summary(self, thread_id: str) -> str:
        data = self._load_payload(thread_id)
        return str(data.get("summary") or "").strip()

    def append_turn(self, thread_id: str, human: str, ai: str) -> None:
        human = (human or "").strip()
        ai = (ai or "").strip()
        if not human:
            return
        data = self._load_payload(thread_id)
        turns = list(data.get("turns") or [])
        turns.append({"human": human, "ai": ai})
        if len(turns) > self._max_turns:
            turns = turns[-self._max_turns :]
        data["turns"] = turns
        self._save_payload(thread_id, data)

    def format_for_prompt(self, thread_id: str) -> str:
        data = self._load_payload(thread_id)
        parts: List[str] = []
        summary = str(data.get("summary") or "").strip()
        if summary:
            parts.append(f"خلاصهٔ گفتگوهای قبلی این نشست:\n{summary}")
        turns = data.get("turns") or []
        if turns:
            lines: List[str] = []
            for item in turns:
                if not isinstance(item, dict):
                    continue
                h = str(item.get("human") or "").strip()
                a = str(item.get("ai") or "").strip()
                if h:
                    lines.append(f"کاربر: {h}")
                if a:
                    lines.append(f"دستیار: {a}")
            if lines:
                parts.append("تاریخچهٔ اخیر:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def clear(self, thread_id: str) -> None:
        if self._redis_ok and self._redis is not None:
            try:
                self._redis.delete(self._redis_key(thread_id))
            except Exception as exc:
                logger.warning("Redis delete failed: %s", exc)
        with self._lock:
            self._local.pop(thread_id, None)
            self._local_summary.pop(thread_id, None)


def build_conversation_memory_from_settings():
    from config.settings import get_settings

    s = get_settings()
    redis_cfg = getattr(s, "redis", None)
    if redis_cfg is not None:
        url = getattr(redis_cfg, "url", "redis://127.0.0.1:6379/0")
        ttl = getattr(redis_cfg, "memory_ttl_seconds", DEFAULT_TTL_SECONDS)
        max_turns = getattr(redis_cfg, "memory_max_turns", DEFAULT_MAX_TURNS)
    else:
        url = "redis://127.0.0.1:6379/0"
        ttl = DEFAULT_TTL_SECONDS
        max_turns = DEFAULT_MAX_TURNS
    return SessionConversationMemory(
        redis_url=url,
        max_turns=max_turns,
        ttl_seconds=ttl,
    )