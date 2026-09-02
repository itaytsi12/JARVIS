"""Jobs: recurring kinds of mission, discovered from the vault.

A Job is not a router rule and not code. It is a Markdown note under
`jobs/` with a `type: job` frontmatter field, and JARVIS finds it the same
way it finds anything else -- by scanning summaries. Dropping

    jobs/write-sales-email.md

into the vault makes that Job available on the next request. Nothing is
registered, imported, or added to a dispatch table; there is no list of
Jobs anywhere in the Python source, deliberately, because a list is the
thing that would have to be edited.

Selection is the same two-stage retrieval everything else uses: the Job's
`summary` and `When To Use` section are what decide whether it applies,
which is why the note format requires them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.index import NoteSummary, VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import JOB, Note
from vault.retrieval import Candidate, VaultRetriever, get_retriever

log = logging.getLogger("jarvis.vault.jobs")

#: The sections a Job note is expected to carry. A Job missing some of
#: them still works -- the missing ones are simply empty -- because a
#: half-written Job the user is still drafting must not be unusable.
JOB_SECTIONS = (
    "Goal",
    "When To Use",
    "Required Context",
    "Required Skills",
    "Procedure",
    "Completion Requirements",
    "Quality Rules",
    "Known Problems",
    "Lessons Learned",
    "Safety / Approval Rules",
)

#: A Job whose status says it is not ready is never selected automatically.
_INACTIVE_STATUSES = frozenset({"placeholder", "draft", "disabled", "retired"})


@dataclass
class Job:
    """One Job note, read in full."""

    note: Note
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.note.title

    @property
    def relative_path(self) -> str:
        return self.note.relative_path

    @property
    def summary(self) -> str:
        return self.note.summary

    @property
    def status(self) -> str:
        return self.note.status

    @property
    def selectable(self) -> bool:
        """Is this Job ready to be chosen automatically?

        The Clipping placeholder is the reason this exists: a Job note may
        legitimately describe work that is not implemented yet, and
        selecting it would produce a confident attempt at something with
        no procedure behind it.
        """
        return self.status not in _INACTIVE_STATUSES

    def section(self, heading: str) -> str:
        return self.note.section(heading)

    @property
    def required_skills(self) -> list[str]:
        """The Skill note titles this Job names, from its wikilinks.

        Plain-text entries are returned too, so a Job written by hand
        without `[[ ]]` still declares its Skills -- they simply will not
        resolve to a note if none exists by that name.
        """
        items = self.note.list_items("Required Skills")
        return [item for item in items if not item.startswith("_")]

    @property
    def procedure(self) -> str:
        return self.section("Procedure")

    @property
    def safety_rules(self) -> str:
        return self.section("Safety / Approval Rules")

    def guidance(self, *, max_chars: int = 3000) -> str:
        """The Job rendered for the model's system prompt.

        Only the sections that actually steer execution -- the goal, the
        procedure, what counts as done, the quality bar, the known
        problems and the safety rules. `When To Use` is a SELECTION
        signal and is deliberately dropped here: the Job has already been
        selected, and repeating the criteria wastes budget.
        """
        parts = [f"## Job: {self.title}", self.note.summary]
        for heading in ("Goal", "Procedure", "Completion Requirements", "Quality Rules", "Known Problems", "Lessons Learned", "Safety / Approval Rules"):
            text = self.section(heading).strip()
            if not text or text.startswith("_Nothing recorded"):
                continue
            parts.append(f"### {heading}\n{text}")
        rendered = "\n\n".join(parts)
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars].rstrip() + "\n... [Job note truncated to fit the context budget]"
        return rendered

    def describe(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "path": self.relative_path,
            "score": round(self.score, 3),
            "status": self.status or "active",
            "required_skills": self.required_skills,
            "why": "; ".join(self.reasons),
        }


class JobRegistry:
    """Discovery and selection over the Job notes currently in the vault."""

    def __init__(self, index: VaultIndex | None = None, vault: VaultManager | None = None, retriever: VaultRetriever | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault
        self.retriever = retriever or get_retriever(index=self.index, vault=self.vault)

    def summaries(self) -> list[NoteSummary]:
        """Every Job's metadata -- the cheap list, used for selection."""
        return self.index.by_type(JOB)

    def titles(self) -> list[str]:
        return sorted(item.title for item in self.summaries())

    def catalog(self) -> str:
        """A compact list of every Job and what it is for.

        Small enough to include in a prompt in full: one line per Job.
        This is what lets the model know a Job exists at all without any
        Job note being read.
        """
        lines = []
        for item in sorted(self.summaries(), key=lambda entry: entry.title.lower()):
            marker = "" if (item.status or "active") not in _INACTIVE_STATUSES else f" [{item.status}]"
            lines.append(f"- {item.title}{marker}: {item.summary}")
        return "\n".join(lines)

    def load(self, title_or_path: str) -> Job | None:
        """One Job by note title, filename or relative path."""
        if not title_or_path:
            return None
        if self.vault.note_exists(title_or_path):
            note = self.vault.read(title_or_path)
            if note is not None and note.note_type == JOB:
                return Job(note=note)
        summary = self.index.find_by_title(title_or_path)
        if summary is None or summary.note_type != JOB:
            return None
        note = self.vault.read(summary.relative_path)
        return Job(note=note) if note is not None else None

    def select(self, request: str, *, min_score: float = 1.2, include_unselectable: bool = False) -> Job | None:
        """The Job that best fits `request`, or None.

        Returning None is a real, expected answer: most requests are not a
        Job at all ("volume down", "what time is it"), and forcing one on
        them would be worse than having none.
        """
        candidates = self.rank(request, include_unselectable=include_unselectable)
        if not candidates:
            return None
        best = candidates[0]
        if best.score < min_score:
            log.debug("No Job selected for %r: best was %s at %.2f", request[:60], best.title, best.score)
            return None
        return best

    def rank(self, request: str, *, include_unselectable: bool = False) -> list[Job]:
        """Every Job that scored at all, best first, with its reasons."""
        scored, _, _ = self.retriever.scan(request, types=[JOB])
        jobs: list[Job] = []
        for candidate in scored:
            note = self.vault.read(candidate.relative_path)
            if note is None:
                continue
            job = Job(note=note, score=candidate.score, reasons=list(candidate.reasons))
            # `When To Use` is the Job's own selection criteria, and it is
            # the one body section worth scoring: a Job whose summary is
            # terse can still say precisely when it applies.
            bonus, why = _when_to_use_bonus(request, job)
            if bonus:
                job.score += bonus
                job.reasons.append(why)
            if not job.selectable and not include_unselectable:
                job.reasons.append(f"not selectable (status={job.status})")
                job.score = 0.0
            jobs.append(job)
        jobs.sort(key=lambda item: (-item.score, item.title))
        return [job for job in jobs if job.score > 0 or include_unselectable]

    def create(
        self,
        title: str,
        *,
        summary: str,
        tags: Iterable[str] = (),
        quick_summary: Iterable[str] = (),
        sections: dict[str, str] | None = None,
    ) -> Job:
        """Author a new Job note that follows the section contract."""
        from vault.paths import JOBS_DIR

        provided = {key.lower(): value for key, value in (sections or {}).items()}
        ordered = [(heading, provided.get(heading.lower(), "")) for heading in JOB_SECTIONS]
        path = self.vault.unique_path(JOBS_DIR, title, fallback="job")
        note = self.vault.create_note(
            path,
            title=title,
            note_type=JOB,
            summary=summary,
            tags=sorted({"job", *[str(tag).lower() for tag in tags]}),
            quick_summary=quick_summary or [summary],
            sections=ordered,
        )
        self.index.invalidate()
        self.index.refresh()
        return Job(note=note)


def _when_to_use_bonus(request: str, job: Job) -> tuple[float, str]:
    from vault.retrieval import _field_score, tokenize

    text = job.section("When To Use")
    if not text:
        return 0.0, ""
    score = _field_score(tokenize(request), text, 1.2)
    return (score, "'When To Use' match") if score else (0.0, "")


_REGISTRY: JobRegistry | None = None


def get_job_registry(index: VaultIndex | None = None, vault: VaultManager | None = None) -> JobRegistry:
    global _REGISTRY
    if index is not None or vault is not None:
        return JobRegistry(index=index, vault=vault)
    if _REGISTRY is None:
        _REGISTRY = JobRegistry()
    return _REGISTRY


def reset_job_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
