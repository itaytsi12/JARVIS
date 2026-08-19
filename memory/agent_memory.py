"""`AgentMemory` -- the single object the rest of JARVIS talks to.

It composes the three stores (conversation, long-term, episodic) and the
retrieval scorer, and enforces the one rule that keeps the model's
context small and useful:

    raw history  ->  extraction  ->  useful memory  ->  ranked retrieval

`observe_exchange` is the write path called after every handled request:
it always appends the raw turns, and promotes a memory only when
`memory.long_term.extract_memories` says something durable was actually
stated. `retrieve` is the read path: a bounded, ranked slice, never the
whole database.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import get_config
from memory.agent_store import AgentDatabase, get_agent_database
from memory.conversation import ASSISTANT, USER, ConversationStore, ConversationTurn
from memory.episodic import Episode, EpisodeStore
from memory.long_term import (
    EXPLICIT,
    FACT,
    LongTermMemoryStore,
    MemoryRecord,
    extract_memories,
)
from memory.retrieval import ScoredEpisode, ScoredMemory, rank_episodes, rank_memories

log = logging.getLogger("jarvis.memory")


@dataclass
class RetrievedContext:
    """Everything memory believes is relevant to one request."""

    query: str
    memories: list[ScoredMemory] = field(default_factory=list)
    episodes: list[ScoredEpisode] = field(default_factory=list)
    recent_turns: list[ConversationTurn] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.memories or self.episodes or self.recent_turns)

    def describe(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "memory_count": len(self.memories),
            "episode_count": len(self.episodes),
            "recent_turn_count": len(self.recent_turns),
            "memory_ids": [item.memory.memory_id for item in self.memories],
            "episode_ids": [item.episode.episode_id for item in self.episodes],
        }


class AgentMemory:
    def __init__(
        self,
        database: AgentDatabase | None = None,
        session_id: str | None = None,
        *,
        conversation: ConversationStore | None = None,
        long_term: LongTermMemoryStore | None = None,
        episodes: EpisodeStore | None = None,
    ):
        database = database or get_agent_database()
        self.conversation = conversation or ConversationStore(database)
        self.long_term = long_term or LongTermMemoryStore(database)
        self.episodes = episodes or EpisodeStore(database)
        self.session_id = self.conversation.start_session(session_id or uuid.uuid4().hex)

    # -- write path ----------------------------------------------------
    def record_user(self, text: str, *, task_id: str | None = None, **metadata: Any) -> ConversationTurn:
        return self.conversation.add_turn(self.session_id, USER, text, task_id=task_id, **metadata)

    def record_assistant(self, text: str, *, task_id: str | None = None, **metadata: Any) -> ConversationTurn:
        return self.conversation.add_turn(self.session_id, ASSISTANT, text, task_id=task_id, **metadata)

    def observe_exchange(
        self,
        user_text: str,
        assistant_text: str | None = None,
        *,
        task_id: str | None = None,
        record_turns: bool = True,
    ) -> list[MemoryRecord]:
        """Store the exchange and promote anything durable it contained.

        Returns only the memories actually written -- an empty list for
        an ordinary command, which is the common and correct case.
        """
        if record_turns:
            self.record_user(user_text, task_id=task_id)
            if assistant_text:
                self.record_assistant(assistant_text, task_id=task_id)
        if not get_config().memory_enabled:
            return []
        candidates = extract_memories(user_text, assistant_text=assistant_text)
        if not candidates:
            return []
        written = self.long_term.remember_candidates(candidates, session_id=self.session_id, task_id=task_id)
        log.info(
            "Long-term memory written: count=%d kinds=%s",
            len(written),
            sorted({record.kind for record in written}),
        )
        return written

    def remember_explicitly(self, text: str, *, kind: str = FACT, importance: int = 5, **metadata: Any) -> MemoryRecord:
        """Store a fact the user directly asked to be remembered."""
        return self.long_term.remember(
            text, kind=kind, importance=importance, source=EXPLICIT, session_id=self.session_id, **metadata
        )

    def record_episode(self, episode: Episode) -> Episode:
        if not episode.session_id:
            episode.session_id = self.session_id
        return self.episodes.record(episode)

    # -- read path -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        max_memories: int | None = None,
        max_episodes: int | None = None,
        recent_turns: int | None = None,
    ) -> RetrievedContext:
        """The bounded, ranked slice of memory relevant to `query`."""
        config = get_config()
        max_memories = config.context_max_memories if max_memories is None else max_memories
        max_episodes = config.context_max_episodes if max_episodes is None else max_episodes
        recent_turns = config.context_recent_turns if recent_turns is None else recent_turns

        memories = rank_memories(query, self.long_term.all(), top_k=max_memories) if max_memories else []
        episodes = rank_episodes(query, self.episodes.recent(60), top_k=max_episodes) if max_episodes else []
        turns = self.conversation.recent_turns(self.session_id, recent_turns) if recent_turns else []
        for item in memories:
            self.long_term.mark_used(item.memory.memory_id)
        return RetrievedContext(query=query, memories=memories, episodes=episodes, recent_turns=turns)

    def search_memories(self, query: str, limit: int = 5) -> list[ScoredMemory]:
        return rank_memories(query, self.long_term.all(), top_k=limit)

    def statistics(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "long_term_memories": self.long_term.count(),
            "conversation_turns": self.conversation.turn_count(),
            "session_turns": self.conversation.turn_count(self.session_id),
            "episodes": self.episodes.statistics(),
        }


_MEMORY: AgentMemory | None = None
_MEMORY_LOCK = threading.Lock()


def get_agent_memory() -> AgentMemory:
    """The process-wide agent memory (one session per JARVIS process)."""
    global _MEMORY
    if _MEMORY is None:
        with _MEMORY_LOCK:
            if _MEMORY is None:
                _MEMORY = AgentMemory()
    return _MEMORY


def reset_agent_memory_for_tests(database: AgentDatabase | None = None) -> AgentMemory:
    global _MEMORY
    with _MEMORY_LOCK:
        _MEMORY = AgentMemory(database)
    return _MEMORY
