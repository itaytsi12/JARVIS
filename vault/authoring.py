"""Creating knowledge for a kind of work JARVIS has never done before.

When a mission-shaped request matches no Job, the wrong outcome is to do
the work and forget how. This module writes the Job note FIRST -- before
the work starts -- so the mission has something to follow and, more
importantly, so the next similar request begins from experience instead
of from nothing.

    "Jarvis, draft a sponsorship proposal for the channel."
      -> no Job matches
      -> create jobs/draft-sponsorship-proposal.md  (draft, with a
         procedure and completion requirements JARVIS can actually state)
      -> create preferences/jobs/draft-sponsorship-proposal.md  (empty)
      -> do the work
      -> record what worked and what failed back into the Job

Two limits keep this from filling the vault with junk:

- **Only mission-shaped requests.** `vault/policy.py` already decides
  that, and "volume down", "mute", "open notepad" and "what's 12 * 8" are
  not it. A Job note for turning the volume down would be pure noise.
- **A draft says it is a draft.** `status: draft` in the frontmatter, and
  an honest Quick Summary. A first-attempt procedure is a guess until a
  mission has actually run it, and the note says so until it has.

Research: some knowledge is only worth writing down with a source
attached -- anything that depends on current information, a platform's
rules, an API, a price or a law. `needs_research` says when, the Job note
gets a `## Sources` section, and `record_source` fills it in as the
mission finds real references. JARVIS does not fabricate citations, and
an unresearched note of that kind stays marked as unverified.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.index import VaultIndex, get_index
from vault.jobs import Job, JobRegistry, get_job_registry
from vault.manager import VaultManager
from vault.note import JOB, Note, extract_section, replace_section
from vault.paths import JOBS_DIR, slugify
from vault.preferences import PreferenceStore, get_preferences

log = logging.getLogger("jarvis.vault.authoring")

#: Frontmatter status for a Job JARVIS wrote for itself and has not yet
#: proven. Distinct from `placeholder`, which means "not implemented".
STATUS_DRAFT = "draft"

#: Subjects whose answer goes stale, where a durable procedure should
#: carry a source. Deliberately narrow: most work ("tidy this folder",
#: "summarise this file") is stable and needs no citation, and demanding
#: one everywhere would just produce fake ones.
_NEEDS_RESEARCH = re.compile(
    r"\b(api|sdk|endpoint|oauth|token|webhook|"
    r"tax|vat|invoice|legal|licen[cs]e|copyright|gdpr|compliance|regulation|"
    r"pricing|price|cost|fee|subscription|billing|"
    r"policy|terms of service|tos|guidelines|"
    r"youtube|tiktok|instagram|twitter|x\.com|linkedin|facebook|reddit|shopify|stripe|paypal|"
    r"latest|current|new version|deprecat\w*|changelog|release notes)\b",
    re.I,
)

#: A verb that names a real kind of work, used to title the Job. Ordered
#: so the most specific reading wins.
_VERBS = (
    "draft", "write", "create", "build", "implement", "design", "produce",
    "generate", "compose", "prepare", "plan", "research", "analyse", "analyze",
    "review", "audit", "summarise", "summarize", "organise", "organize",
    "configure", "set up", "install", "deploy", "publish", "post", "send",
    "convert", "export", "import", "migrate", "refactor", "optimise", "optimize",
    "clean up", "tidy", "back up", "schedule", "book", "order", "compare",
)

_STOPWORDS = frozenset(
    "the a an my our your this that these those for me please jarvis hey sir "
    "can could would you i we to of in on at and or but with by from now today".split()
)


def needs_research(request: str) -> bool:
    """Should a Job for this work carry sources before it is trusted?"""
    return bool(_NEEDS_RESEARCH.search(request or ""))


def propose_job_title(request: str) -> str:
    """A stable, reusable name for this KIND of work.

    The title has to generalise: "Draft Sponsorship Proposal", not "Draft
    a sponsorship proposal for the channel tonight". A Job named after one
    request would never match the next one, which is the entire point of
    writing it down.
    """
    text = re.sub(r"\s+", " ", (request or "").strip())
    text = re.sub(r"^(?:hey\s+)?jarvis[,\s]+", "", text, flags=re.I)
    text = re.sub(r"^(?:please|could you|can you|i want you to|i need you to|would you)\s+", "", text, flags=re.I)
    lowered = text.lower()

    verb = next((item for item in _VERBS if re.search(rf"\b{re.escape(item)}\b", lowered)), "")
    if verb:
        after = lowered.split(verb, 1)[1]
    else:
        after = lowered
        verb = "handle"

    words: list[str] = []
    for word in re.findall(r"[a-z0-9][a-z0-9'-]*", after):
        if word in _STOPWORDS:
            if words:
                continue
            continue
        words.append(word)
        if len(words) >= 3:
            break
    parts = [verb, *words] if words else [verb, "request"]
    title = " ".join(parts).strip()
    return " ".join(word.capitalize() if len(word) > 2 else word for word in title.split()).strip() or "General Task"


@dataclass
class AuthoredJob:
    """A Job note JARVIS wrote for itself."""

    job: Job
    preference_path: str = ""
    researched: bool = False
    created: bool = True
    reason: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "title": self.job.title if self.job else None,
            "path": self.job.relative_path if self.job else None,
            "preferences": self.preference_path,
            "needs_research": self.researched,
            "created": self.created,
            "reason": self.reason,
        }


class JobAuthor:
    """Writes a first Job note, then improves it from what happened."""

    def __init__(
        self,
        vault: VaultManager | None = None,
        index: VaultIndex | None = None,
        jobs: JobRegistry | None = None,
        preferences: PreferenceStore | None = None,
    ):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault
        self.jobs = jobs or get_job_registry(index=self.index, vault=self.vault)
        self.preferences = preferences or get_preferences(vault=self.vault, index=self.index)

    def create_for(self, request: str, *, skills: Iterable[str] = ()) -> AuthoredJob | None:
        """Author a draft Job for a kind of work with no Job yet.

        Returns None when a Job with this name already exists -- the
        second occurrence of a kind of work must IMPROVE the note, not
        create a rival one beside it.
        """
        title = propose_job_title(request)
        existing = self.jobs.load(title)
        if existing is not None:
            return AuthoredJob(job=existing, created=False, reason="a Job of this kind already exists")

        research = needs_research(request)
        skill_list = [str(item) for item in skills if str(item).strip()]
        sections = {
            "Goal": f"Complete work of this kind: {request.strip()[:200]}",
            "When To Use": (
                f"The user asks for work of this kind (for example: \"{request.strip()[:120]}\"). "
                "Refine this line as real examples accumulate -- it is what decides whether this Job "
                "is selected at all."
            ),
            "Required Context": "- Whatever the specific request names.\n- The relevant project note, if one exists.",
            "Required Skills": "\n".join(f"- [[{name}]]" for name in skill_list) or "_None identified yet._",
            "Procedure": (
                "1. Restate what is actually being asked, and confirm anything genuinely ambiguous.\n"
                "2. Gather what the work needs"
                + (" -- and, because this subject changes, check a current, reliable source first.\n" if research else ".\n")
                + "3. Do the work in the smallest sequence of steps that achieves it.\n"
                "4. Verify the result against what was asked, by observation rather than assumption.\n"
                "5. Report the result.\n\n"
                "_This procedure is a first draft, written before this Job had ever run. It is "
                "replaced by what actually works as missions complete._"
            ),
            "Completion Requirements": "- The thing that was asked for exists, and JARVIS has observed that it does.",
            "Quality Rules": "- Never claim an outcome that was not observed.",
            "Known Problems": "_Nothing recorded yet._",
            "Lessons Learned": "_Nothing recorded yet._",
            "Safety / Approval Rules": "- See [[Protected Rules]].",
        }
        if research:
            sections["Sources"] = (
                "_No sources recorded yet._ This Job depends on information that changes, so its "
                "procedure is not durable knowledge until a real, current source is recorded here."
            )

        path = self.vault.unique_path(JOBS_DIR, title, fallback="job")
        note = self.vault.create_note(
            path,
            title=title,
            note_type=JOB,
            summary=f"Draft procedure for work of this kind: {request.strip()[:150]}",
            tags=sorted({"job", "draft", *[slugify(word) for word in title.lower().split()[:3] if len(word) > 3]}),
            quick_summary=[
                "DRAFT. Written by JARVIS when this kind of work first came up, and not yet proven.",
                f"First request that produced it: \"{request.strip()[:120]}\"",
                (
                    "Depends on information that changes -- verify against a current source before trusting it."
                    if research
                    else "Improved by each mission that uses it."
                ),
            ],
            sections=list(sections.items()),
            extra_metadata={"status": STATUS_DRAFT, "authored_by": "jarvis", "needs_research": research},
        )
        self.preferences.ensure_job(title, job_path=path)
        self.preferences.link_job(path, title)
        self.index.invalidate()
        self.index.refresh()
        log.info("Authored a draft Job for a new kind of work: %s (%s)", title, path)
        return AuthoredJob(
            job=Job(note=note),
            preference_path=f"preferences/jobs/{slugify(title)}.md",
            researched=research,
            reason="no existing Job covered this kind of work",
        )

    # ------------------------------------------------------- improvement
    def record_outcome(
        self,
        job_title: str,
        *,
        succeeded: bool,
        worked: str = "",
        failed: Iterable[str] = (),
        verified: bool = False,
    ) -> bool:
        """Fold what happened back into the Job note.

        A draft that has now succeeded AND been verified stops being a
        draft -- that is the only thing that promotes it, because a
        procedure nobody has run is a guess however plausible it reads.
        """
        job = self.jobs.load(job_title)
        if job is None:
            return False
        path = job.relative_path
        failures = [str(item).strip() for item in failed if str(item).strip()]

        def mutate(note: Note) -> Note:
            if worked.strip():
                section = extract_section(note.body, "Lessons Learned")
                entry = f"- {worked.strip()}"
                if entry not in section:
                    merged = entry if not section.strip() or section.startswith("_Nothing") else f"{section.rstrip()}\n{entry}"
                    note.body = replace_section(note.body, "Lessons Learned", merged)
            if failures:
                section = extract_section(note.body, "Known Problems")
                lines = [f"- {item}" for item in failures if item not in section]
                if lines:
                    merged = "\n".join(lines) if not section.strip() or section.startswith("_Nothing") else f"{section.rstrip()}\n" + "\n".join(lines)
                    note.body = replace_section(note.body, "Known Problems", merged)
            if succeeded and verified and str(note.metadata.get("status") or "") == STATUS_DRAFT:
                metadata = dict(note.metadata)
                metadata.pop("status", None)
                metadata["proven"] = True
                note.metadata = metadata
                note.body = note.body.replace(
                    "_This procedure is a first draft, written before this Job had ever run. It is "
                    "replaced by what actually works as missions complete._",
                    "_Proven: a mission has completed this procedure and verified the result._",
                )
                quick = note.quick_summary.replace(
                    "DRAFT. Written by JARVIS when this kind of work first came up, and not yet proven.",
                    "Written by JARVIS the first time this work came up, and since proven by a verified mission.",
                )
                note.body = replace_section(note.body, "Quick Summary", quick)
            return note

        updated = self.vault.update_note(path, mutate)
        self.index.invalidate()
        self.index.refresh()
        if updated is not None:
            log.info("Improved authored Job %s from a %s mission", job_title, "successful" if succeeded else "failed")
        return updated is not None

    def record_source(self, job_title: str, *, url: str, note_text: str = "") -> bool:
        """Attach a real source to a Job whose knowledge goes stale."""
        job = self.jobs.load(job_title)
        if job is None or not url.strip():
            return False
        entry = f"- {url.strip()}" + (f" -- {note_text.strip()}" if note_text.strip() else "")

        def mutate(note: Note) -> Note:
            section = extract_section(note.body, "Sources")
            if url.strip() in section:
                return note
            merged = entry if not section.strip() or section.startswith("_No sources") else f"{section.rstrip()}\n{entry}"
            note.body = replace_section(note.body, "Sources", merged)
            return note

        updated = self.vault.update_note(job.relative_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        return updated is not None


_AUTHOR: JobAuthor | None = None


def get_job_author(vault: VaultManager | None = None, index: VaultIndex | None = None) -> JobAuthor:
    global _AUTHOR
    if vault is not None or index is not None:
        return JobAuthor(vault=vault, index=index)
    if _AUTHOR is None:
        _AUTHOR = JobAuthor()
    return _AUTHOR


def reset_job_author() -> None:
    global _AUTHOR
    _AUTHOR = None
