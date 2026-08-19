"""Relevance-based retrieval over long-term memories and episodes.

Scoring combines three signals, all computed locally:

- **overlap**: Jaccard-style token overlap between the query and the
  record, with a small boost for rarer, longer tokens so a distinctive
  word ("elevenlabs") counts for more than "the";
- **recency**: a gentle exponential decay, used to separate several
  plausible matches -- never to surface a record that matches nothing
  (an episode with zero lexical overlap is dropped outright);
- **importance / outcome**: an explicit memory outweighs an inferred
  one; a verified successful episode outweighs a failed one.

This is deliberately dependency-free rather than an embedding index.
`score_text` is the only place similarity is computed, so swapping in a
vector backend later means replacing one function, not the retrieval API.
`EMBEDDING_BACKEND` is the hook a future implementation registers into.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from memory.episodic import Episode
from memory.long_term import EXPLICIT, MemoryRecord

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those is are was were be been being
    do does did doing have has had having i me my we our you your it its of in on at to for
    with from by as so not no yes can could would should will just about into over under
    please jarvis sir""".split()
)

# A future embedding retriever registers itself here; when set, it is used
# INSTEAD of token overlap. Nothing else in this module changes.
EMBEDDING_BACKEND: Callable[[str, str], float] | None = None


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1}


def score_text(query: str, candidate: str) -> float:
    """Similarity in [0, 1] between a query and one candidate text."""
    if EMBEDDING_BACKEND is not None:
        try:
            return max(0.0, min(1.0, float(EMBEDDING_BACKEND(query, candidate))))
        except Exception:
            pass  # a broken optional backend must never break retrieval
    query_tokens = tokenize(query)
    candidate_tokens = tokenize(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    shared = query_tokens & candidate_tokens
    if not shared:
        return 0.0
    # Longer shared tokens are more discriminating than short common ones.
    weight = sum(1.0 + min(len(token), 12) / 24 for token in shared)
    return min(1.0, weight / (len(query_tokens) + 0.5 * len(candidate_tokens - query_tokens)))


def recency_score(timestamp: str | None, half_life_hours: float = 72.0) -> float:
    if not timestamp:
        return 0.0
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0.0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (datetime.now(timezone.utc) - moment).total_seconds() / 3600)
    return math.exp(-age_hours / max(half_life_hours, 0.1))


@dataclass
class ScoredMemory:
    memory: MemoryRecord
    score: float
    reason: str = ""


@dataclass
class ScoredEpisode:
    episode: Episode
    score: float
    reason: str = ""


def rank_memories(
    query: str,
    memories: Iterable[MemoryRecord],
    *,
    top_k: int = 5,
    min_score: float = 0.08,
) -> list[ScoredMemory]:
    ranked: list[ScoredMemory] = []
    for memory in memories:
        overlap = score_text(query, f"{memory.text} {memory.subject or ''}")
        importance = min(memory.importance, 5) / 5
        explicit_bonus = 0.15 if memory.source == EXPLICIT else 0.0
        score = 0.65 * overlap + 0.2 * importance + explicit_bonus + 0.05 * recency_score(memory.updated_at, 336.0)
        # An explicitly stated fact stays retrievable even when the
        # wording of a later request shares nothing with it -- that is
        # the whole reason the user bothered to state it.
        if memory.source == EXPLICIT and overlap == 0.0:
            score = max(score, min_score)
        if score >= min_score:
            ranked.append(
                ScoredMemory(
                    memory,
                    round(score, 4),
                    reason=f"overlap={overlap:.2f} importance={memory.importance} source={memory.source}",
                )
            )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: max(0, top_k)]


def rank_episodes(
    query: str,
    episodes: Iterable[Episode],
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> list[ScoredEpisode]:
    ranked: list[ScoredEpisode] = []
    for episode in episodes:
        overlap = score_text(query, episode.searchable_text())
        if overlap <= 0.0:
            # An episode with nothing lexically in common with the request
            # is not relevant, however recent or successful it was.
            # Outcome and recency are tie-breakers between plausible
            # episodes, never a reason to surface an unrelated one.
            continue
        outcome = 0.15 if episode.verified else 0.08 if episode.success else 0.0
        score = 0.7 * overlap + outcome + 0.15 * recency_score(episode.created_at)
        if score >= min_score:
            ranked.append(
                ScoredEpisode(
                    episode,
                    round(score, 4),
                    reason=f"overlap={overlap:.2f} success={episode.success} verified={episode.verified}",
                )
            )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: max(0, top_k)]


def describe_ranking(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Compact, log-safe view of a ranking, for observability."""
    described = []
    for item in items:
        if isinstance(item, ScoredMemory):
            described.append({"type": "memory", "id": item.memory.memory_id, "score": item.score, "reason": item.reason})
        elif isinstance(item, ScoredEpisode):
            described.append({"type": "episode", "id": item.episode.episode_id, "score": item.score, "reason": item.reason})
    return described
