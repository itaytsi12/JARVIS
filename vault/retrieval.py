"""Two-stage retrieval: scan every summary, deep-read almost none.

    Mission: "Fix the Apple Music playlist problem."

    STAGE 1 -- scan (cheap, offline, no model call)
      Apple Music Control   how JARVIS controls Apple Music        score 0.81
      Spotify               how JARVIS controls Spotify            score 0.22
      Video Editing         short-form video processing            score 0.00
      Python Debugging      debugging Python projects              score 0.11

    STAGE 2 -- deep read (bounded by a character budget)
      skills/apple-music-control.md
      projects/jarvis.md

Only those two notes' full text ever reaches the model. The other
summaries were seen and rejected for a fraction of the cost, and the
`RetrievalTrace` records that they were seen -- so "why did JARVIS not
use note X" is answerable from a log rather than by guessing.

Ranking is deterministic and free: no embedding model, no model call, no
network. It is lexical overlap over the fields the note format
guarantees, weighted by where a term matched (a hit in the title means
more than a hit in a tag), plus small structural bonuses (an explicitly
requested type, a recently updated note, an explicitly linked note). That
is enough because the note format does the hard part: every note already
carries a hand-written one-sentence statement of what it is for.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from vault.index import NoteSummary, VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import INDEX, Note

log = logging.getLogger("jarvis.vault.retrieval")

#: Words that carry no selection signal. Kept deliberately small -- an
#: over-eager stop list throws away real terms ("state", "system", "run"
#: are all meaningful in this vault).
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here it its it's
    is are was were be been being am do does did doing done have has had having
    i me my we our you your he she they them his her their of in on at to for
    from with by about as into over under again further once no not only own same
    so too very can will just should now what which who whom whose when where why how
    please jarvis sir okay ok yeah yes hey
    """.split()
)

_WORD = re.compile(r"[A-Za-z0-9_֐-׿][A-Za-z0-9_'֐-׿-]*")

#: Where a term matched, and what that is worth. A title hit is the
#: strongest signal the format offers; a body hit is not available at all
#: at this stage, which is the point.
_WEIGHT_TITLE = 3.0
_WEIGHT_SUMMARY = 2.0
_WEIGHT_TAGS = 1.6
_WEIGHT_QUICK = 1.0
_WEIGHT_PATH = 0.6


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def _field_score(terms: Sequence[str], text: str, weight: float) -> float:
    if not text or not terms:
        return 0.0
    haystack = set(tokenize(text))
    if not haystack:
        return 0.0
    hits = 0.0
    for term in terms:
        if term in haystack:
            hits += 1.0
            continue
        # A stemless partial match: "playlists" against "playlist",
        # "debugging" against "debug". Worth less than an exact hit, and
        # bounded to terms long enough not to match everything.
        if len(term) >= 5 and any(word.startswith(term[:5]) for word in haystack):
            hits += 0.5
    return weight * hits


@dataclass
class Candidate:
    """One scanned note and why it did or did not make the cut."""

    summary: NoteSummary
    score: float
    reasons: list[str] = field(default_factory=list)
    selected: bool = False

    @property
    def relative_path(self) -> str:
        return self.summary.relative_path

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.summary.relative_path,
            "title": self.summary.title,
            "type": self.summary.note_type,
            "score": round(self.score, 3),
            "selected": self.selected,
            "why": "; ".join(self.reasons),
        }


@dataclass
class RetrievalTrace:
    """What the scan saw, what it chose, and why.

    This is a high-level selection rationale -- note titles, scores and
    the reason a note ranked -- and deliberately not the model's internal
    reasoning, which this system never captures.
    """

    query: str
    scanned: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    deep_read_chars: int = 0
    budget_chars: int = 0
    scan_ms: float = 0.0
    read_ms: float = 0.0
    dropped_for_budget: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "query": self.query[:120],
            "scanned_notes": self.scanned,
            "considered": [item.describe() for item in self.candidates[:12]],
            "selected": list(self.selected),
            "dropped_for_budget": list(self.dropped_for_budget),
            "deep_read_chars": self.deep_read_chars,
            "budget_chars": self.budget_chars,
            "scan_ms": round(self.scan_ms, 2),
            "read_ms": round(self.read_ms, 2),
        }

    def explain(self) -> str:
        """The human-readable version, for the log and for `--vault-explain`."""
        lines = [f'Mission: "{self.query[:120]}"', f"Scanned {self.scanned} note summaries in {self.scan_ms:.1f}ms."]
        if self.candidates:
            lines.append("Candidates considered:")
            for candidate in self.candidates[:8]:
                mark = "*" if candidate.selected else " "
                lines.append(f"  {mark} {candidate.summary.title} ({candidate.score:.2f}) -- {'; '.join(candidate.reasons) or 'no signal'}")
        lines.append(
            "Selected for full read: " + (", ".join(self.selected) if self.selected else "(nothing scored high enough)")
        )
        if self.dropped_for_budget:
            lines.append("Dropped for context budget: " + ", ".join(self.dropped_for_budget))
        lines.append(f"Deep-read {self.deep_read_chars} of {self.budget_chars} budgeted characters.")
        return "\n".join(lines)


@dataclass
class Retrieved:
    """The result of a two-stage retrieval: full notes plus the trace."""

    notes: list[Note] = field(default_factory=list)
    trace: RetrievalTrace | None = None

    def paths(self) -> list[str]:
        return [note.relative_path for note in self.notes]

    def text(self) -> str:
        blocks = []
        for note in self.notes:
            blocks.append(f"### {note.title} ({note.relative_path})\n{note.body.strip()}")
        return "\n\n".join(blocks)


class VaultRetriever:
    """Scan, rank, select, deep-read."""

    def __init__(self, index: VaultIndex | None = None, vault: VaultManager | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    # ----------------------------------------------------------- stage 1
    def scan(
        self,
        query: str,
        *,
        types: Iterable[str] | None = None,
        exclude_types: Iterable[str] = (),
        tags: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
        include_index_notes: bool = False,
        boost_paths: Iterable[str] = (),
    ) -> tuple[list[Candidate], int, float]:
        """Rank every note's METADATA against `query`. Nothing is read."""
        started = time.perf_counter()
        summaries = self.index.summaries()
        wanted_types = {str(item).lower() for item in types} if types else None
        unwanted_types = {str(item).lower() for item in exclude_types}
        wanted_tags = {str(item).lower() for item in tags} if tags else None
        excluded = set(exclude)
        boosted = {str(path) for path in boost_paths}
        terms = tokenize(query)
        now = time.time()

        candidates: list[Candidate] = []
        for item in summaries:
            if item.relative_path in excluded:
                continue
            if item.note_type == INDEX and not include_index_notes:
                continue
            if wanted_types is not None and item.note_type not in wanted_types:
                continue
            if item.note_type in unwanted_types:
                continue
            if wanted_tags is not None and not (wanted_tags & {tag.lower() for tag in item.tags}):
                continue

            reasons: list[str] = []
            title_score = _field_score(terms, item.title, _WEIGHT_TITLE)
            summary_score = _field_score(terms, item.summary, _WEIGHT_SUMMARY)
            tag_score = _field_score(terms, " ".join(item.tags), _WEIGHT_TAGS)
            quick_score = _field_score(terms, item.quick_summary, _WEIGHT_QUICK)
            path_score = _field_score(terms, item.relative_path.replace("/", " ").replace("-", " "), _WEIGHT_PATH)

            #: The TOPICAL score: how well this note's own words match the
            #: request. Everything below is a structural bonus, and a
            #: structural bonus may only REORDER notes that already have a
            #: topical signal -- it may never qualify one that has none.
            #: Getting this wrong is not theoretical: "requested type"
            #: (0.75) plus "recently touched" (0.4) came to 1.15 and
            #: cleared a 1.0 threshold on its own, which loaded a Video
            #: Editing skill into an Apple Music mission.
            topical = title_score + summary_score + tag_score + quick_score + path_score
            if title_score:
                reasons.append("title match")
            if summary_score:
                reasons.append("summary match")
            if tag_score:
                reasons.append("tag match")
            if quick_score:
                reasons.append("quick-summary match")
            if path_score and not reasons:
                reasons.append("path match")

            score = topical
            if item.relative_path in boosted:
                # The one bonus that stands alone: an explicit link from a
                # note already selected is a statement by the vault itself
                # that the two belong together, not an inference.
                score += 2.5
                reasons.append("linked from an already-selected note")

            if topical > 0:
                if wanted_types is not None and item.note_type in wanted_types:
                    score += 0.75
                    reasons.append(f"requested type: {item.note_type}")
                if item.status == "active":
                    score += 0.5
                    reasons.append("active")
                # Recency, gently: a note touched today is more likely to
                # be the one in play than one untouched for a year -- but
                # it must never outrank, or substitute for, a real match.
                if item.mtime:
                    age_days = max(0.0, (now - item.mtime) / 86400.0)
                    score += 0.4 * math.exp(-age_days / 14.0)

            if score > 0:
                candidates.append(Candidate(summary=item, score=score, reasons=reasons))

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.summary.relative_path))
        return candidates, len(summaries), (time.perf_counter() - started) * 1000

    # ----------------------------------------------------------- stage 2
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        budget_chars: int = 8000,
        min_score: float = 1.0,
        types: Iterable[str] | None = None,
        exclude_types: Iterable[str] = (),
        tags: Iterable[str] | None = None,
        always: Iterable[str] = (),
        exclude: Iterable[str] = (),
        follow_links: bool = True,
    ) -> Retrieved:
        """Scan, then deep-read at most `limit` notes within the budget.

        `always` names notes that are read regardless of score (identity,
        core rules, the active mission) -- they are what makes JARVIS
        itself, not a search result.
        """
        candidates, scanned, scan_ms = self.scan(query, types=types, exclude_types=exclude_types, tags=tags, exclude=exclude)
        trace = RetrievalTrace(query=query, scanned=scanned, candidates=candidates[:20], scan_ms=scan_ms, budget_chars=budget_chars)

        chosen: list[str] = []
        for path in always:
            if path and path not in chosen and self.vault.note_exists(path):
                chosen.append(path)
        for candidate in candidates:
            if len(chosen) >= limit + len(always):
                break
            if candidate.score < min_score:
                continue
            if candidate.relative_path in chosen:
                continue
            candidate.selected = True
            chosen.append(candidate.relative_path)

        # One hop along the wikilink graph: a Job that names its required
        # Skills must pull those Skills in, which is exactly what
        # `Required Skills: [[Python Debugging]]` is FOR. Strictly one hop
        # -- an unbounded graph walk is how "load the relevant notes"
        # silently becomes "load the vault".
        if follow_links and chosen:
            linked = self._linked_paths(chosen)
            for path in linked:
                if len(chosen) >= limit + len(always) + 3:
                    break
                if path not in chosen:
                    chosen.append(path)
                    summary = self.index.get(path)
                    if summary is not None:
                        trace.candidates.append(
                            Candidate(summary=summary, score=0.0, reasons=["linked from a selected note"], selected=True)
                        )

        read_started = time.perf_counter()
        notes: list[Note] = []
        used = 0
        for path in chosen:
            note = self.vault.read(path)
            if note is None:
                continue
            cost = len(note.body)
            if used + cost > budget_chars and notes:
                trace.dropped_for_budget.append(path)
                continue
            if used + cost > budget_chars:
                # The very first note alone exceeds the budget: truncate
                # it rather than returning nothing at all, and say so. The
                # notice is part of what is written, so it comes OUT of the
                # remaining budget rather than being added on top of it --
                # otherwise "enforced" is off by the length of the notice.
                notice = "\n\n_[truncated to fit the context budget]_"
                room = max(0, budget_chars - used - len(notice))
                note.body = note.body[:room].rstrip() + notice
                cost = len(note.body)
            notes.append(note)
            used += cost
        trace.read_ms = (time.perf_counter() - read_started) * 1000
        trace.selected = [note.relative_path for note in notes]
        trace.deep_read_chars = used

        log.debug("Vault retrieval: %s", trace.describe())
        return Retrieved(notes=notes, trace=trace)

    def _linked_paths(self, paths: Iterable[str]) -> list[str]:
        found: list[str] = []
        for path in paths:
            summary = self.index.get(path)
            if summary is None:
                note = self.vault.read(path)
                if note is None:
                    continue
                links = note.links
            else:
                links = summary.links
            for title in links:
                target = self.index.find_by_title(title, refresh=False)
                if target is not None and target.relative_path not in found:
                    found.append(target.relative_path)
        return found


_RETRIEVER: VaultRetriever | None = None


def get_retriever(index: VaultIndex | None = None, vault: VaultManager | None = None) -> VaultRetriever:
    global _RETRIEVER
    if index is not None or vault is not None:
        return VaultRetriever(index=index, vault=vault)
    if _RETRIEVER is None:
        _RETRIEVER = VaultRetriever()
    return _RETRIEVER


def reset_retriever() -> None:
    global _RETRIEVER
    _RETRIEVER = None
