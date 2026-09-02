"""Preferences, in two levels: global, and one note per Job.

    preferences/
      global.md                  applies to every mission
      jobs/
        fix-software-bug.md      applies only to that Job
        clipping.md
        ...

Two rules make this work, and both are about what does NOT happen.

**Preference notes are not scanned.** There is one per Job, so they grow
with the Jobs; scanning them would spend the retrieval budget on notes
that say how to do a job rather than when it applies, and they would
compete with the Jobs themselves. Instead:

- the GLOBAL note is loaded by policy, for every full mission, because it
  must apply whether or not its words happen to match the request;
- a JOB note is loaded only AFTER its Job has been selected, through the
  explicit `## Preferences` reference in the Job note.

`vault/paths.py::exclusion_reason` and `VaultIndex.refresh` enforce the
exclusion in one place. Resolving `[[Preferences - Fix Software Bug]]` is
still possible, because naming one note is a reference, not a search.

**Job preferences win over global ones.** When the two conflict, the Job's
rule applies inside that Job and nowhere else. That is the whole conflict
model -- there is no inheritance chain, no priority arithmetic, and no
merge language beyond "the Job said otherwise".

Preference notes start empty and stay lean. They grow from what the user
actually SAYS -- "always", "never", "next time", "I prefer", "don't do
that again" -- and never from JARVIS inferring a rule because something
went well once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.archive import KIND_PREFERENCE, get_archive
from vault.consolidation import DUPLICATE, REFINE, SUPERSEDE, classify_against, merge_scoped
from vault.index import VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import Note, extract_list_items, extract_section, replace_section
from vault.paths import (
    GLOBAL_PREFERENCES_NOTE,
    JOB_PREFERENCES_DIR,
    slugify,
)

log = logging.getLogger("jarvis.vault.preferences")

#: The section a preference note keeps its rules in.
SECTION_RULES = "Preferences"
#: The section a Job note names its preference note in.
JOB_PREFERENCES_SECTION = "Preferences"

#: What an empty Job preference note says. Deliberately a sentence rather
#: than an invented rule: JARVIS must not guess preferences for the user.
EMPTY_MARKER = "_No Job-specific preferences recorded yet._"


def job_preference_path(job_title: str) -> str:
    return f"{JOB_PREFERENCES_DIR}/{slugify(job_title, fallback='job')}.md"


def job_preference_title(job_title: str) -> str:
    """The note TITLE, which is what a Job's wikilink points at."""
    return f"Preferences - {job_title}"


@dataclass
class ResolvedPreferences:
    """The preferences in force for one mission."""

    global_rules: list[str] = field(default_factory=list)
    job_rules: list[str] = field(default_factory=list)
    job_title: str = ""
    #: Global rules the Job's own rules override, kept so the decision is
    #: observable rather than a silent disappearance.
    overridden: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    @property
    def effective(self) -> list[str]:
        """Global rules minus the ones this Job overrides, then the Job's."""
        kept = [rule for rule in self.global_rules if rule not in self.overridden]
        return kept + self.job_rules

    def render(self, *, max_chars: int = 1400) -> str:
        if not self.effective:
            return ""
        parts = ["## How the user wants this done"]
        kept = [rule for rule in self.global_rules if rule not in self.overridden]
        if kept:
            parts.append("\n".join(f"- {rule}" for rule in kept))
        if self.job_rules:
            parts.append(
                f"### For {self.job_title or 'this job'} specifically (these override the general rules)\n"
                + "\n".join(f"- {rule}" for rule in self.job_rules)
            )
        rendered = "\n\n".join(parts)
        return rendered[:max_chars] if len(rendered) > max_chars else rendered

    def describe(self) -> dict[str, Any]:
        return {
            "job": self.job_title or None,
            "global_rules": len(self.global_rules),
            "job_rules": len(self.job_rules),
            "overridden": list(self.overridden),
            "paths": list(self.paths),
        }


class PreferenceStore:
    """Read, write and resolve the two preference levels."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault

    # ------------------------------------------------------------ read
    def global_note(self) -> Note | None:
        return self.vault.read(GLOBAL_PREFERENCES_NOTE)

    def global_rules(self) -> list[str]:
        note = self.global_note()
        if note is None:
            return []
        return [rule for rule in extract_list_items(note.section(SECTION_RULES)) if not rule.startswith("_")]

    def job_note(self, job_title: str) -> Note | None:
        return self.vault.read(job_preference_path(job_title))

    def job_rules(self, job_title: str) -> list[str]:
        note = self.job_note(job_title)
        if note is None:
            return []
        return [rule for rule in extract_list_items(note.section(SECTION_RULES)) if not rule.startswith("_")]

    def resolve(self, job_title: str = "") -> ResolvedPreferences:
        """Global rules plus this Job's, with the Job winning on conflict."""
        resolved = ResolvedPreferences(global_rules=self.global_rules(), job_title=job_title)
        if self.vault.note_exists(GLOBAL_PREFERENCES_NOTE):
            resolved.paths.append(GLOBAL_PREFERENCES_NOTE)
        if not job_title:
            return resolved
        resolved.job_rules = self.job_rules(job_title)
        if resolved.job_rules and self.vault.note_exists(job_preference_path(job_title)):
            resolved.paths.append(job_preference_path(job_title))
        # The whole conflict model: a global rule the Job contradicts is
        # dropped for this Job. No priority arithmetic, no merging.
        for rule in resolved.global_rules:
            for job_rule in resolved.job_rules:
                action, _, _ = classify_against([rule], job_rule)
                if action in {SUPERSEDE, REFINE, DUPLICATE}:
                    resolved.overridden.append(rule)
                    break
        return resolved

    # ----------------------------------------------------------- write
    def ensure_global(self) -> Note:
        if self.vault.note_exists(GLOBAL_PREFERENCES_NOTE):
            return self.vault.read(GLOBAL_PREFERENCES_NOTE)  # type: ignore[return-value]
        note = self.vault.create_note(
            GLOBAL_PREFERENCES_NOTE,
            title="Global Preferences",
            note_type="user",
            summary="How the user wants JARVIS to behave on every job, unless a Job's own preferences say otherwise.",
            tags=["preferences", "global", "user"],
            quick_summary=[
                "These apply to every mission.",
                "A Job's own preference note overrides any of these inside that Job.",
                "Only what the user actually stated goes here -- never an inferred rule.",
            ],
            sections=[(SECTION_RULES, ""), ("Notes", "")],
        )
        self.index.invalidate()
        self.index.refresh()
        return note

    def ensure_job(self, job_title: str, *, job_path: str = "") -> Note:
        """Create this Job's preference note if it has none.

        It starts explicitly EMPTY. Inventing a starting set of
        preferences would put words in the user's mouth, and every one of
        them would then quietly steer the Job.
        """
        path = job_preference_path(job_title)
        if self.vault.note_exists(path):
            return self.vault.read(path)  # type: ignore[return-value]
        note = self.vault.create_note(
            path,
            title=job_preference_title(job_title),
            note_type="user",
            summary=f"How the user wants the {job_title} job done. Overrides the global preferences inside this job.",
            tags=["preferences", "job", slugify(job_title)],
            quick_summary=[
                f"Applies only to [[{job_title}]].",
                "Loaded after that Job is selected, never through a vault-wide scan.",
                "Overrides a conflicting global preference inside this Job.",
            ],
            sections=[(SECTION_RULES, EMPTY_MARKER), ("Notes", "")],
            extra_metadata={"job": job_title, "applies_to": job_path or "", "scan": "excluded"},
        )
        self.index.invalidate()
        self.index.refresh()
        log.info("Created a preference note for Job %s", job_title)
        return note

    def record(
        self,
        rule: str,
        *,
        job_title: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Write one durable preference, archiving whatever it replaces.

        A rule that contradicts an existing one supersedes it: the new
        rule takes its place in the active note and the old one moves to
        the archive with the date and the reason. A rule that merely
        scopes an existing one refines it into a single sentence. Neither
        outcome loses the old wording -- the archive keeps it.
        """
        rule = (rule or "").strip()
        if not rule:
            return {"applied": False, "reason": "nothing to record"}

        if job_title:
            self.ensure_job(job_title)
            path = job_preference_path(job_title)
            scope = f"the {job_title} job"
        else:
            self.ensure_global()
            path = GLOBAL_PREFERENCES_NOTE
            scope = "every job"

        note = self.vault.read(path)
        if note is None:
            return {"applied": False, "reason": f"{path} could not be read"}

        existing = [item for item in extract_list_items(note.section(SECTION_RULES)) if not item.startswith("_")]
        action, target, why = classify_against(existing, rule)

        if action == DUPLICATE:
            return {"applied": False, "action": DUPLICATE, "path": path, "rule": target, "reason": why}

        replacement = merge_scoped(target, rule) if action == REFINE else rule
        if action in {SUPERSEDE, REFINE}:
            get_archive(vault=self.vault, index=self.index).archive_rule(
                kind=KIND_PREFERENCE,
                text=target,
                source_path=path,
                reason=reason or why,
                replaced_by=replacement,
            )

        def mutate(target_note: Note) -> Note:
            current = extract_section(target_note.body, SECTION_RULES)
            lines = [line for line in current.splitlines() if line.strip() and line.strip() != EMPTY_MARKER]
            if action in {SUPERSEDE, REFINE} and target:
                for position, line in enumerate(lines):
                    if target in line:
                        lines[position] = f"- {replacement}"
                        break
                else:
                    lines.append(f"- {replacement}")
            else:
                lines.append(f"- {replacement}")
            target_note.body = replace_section(target_note.body, SECTION_RULES, "\n".join(lines))
            return target_note

        updated = self.vault.update_note(path, mutate)
        self.index.invalidate()
        self.index.refresh()
        log.info("Preference recorded for %s: %s", scope, replacement)
        return {
            "applied": updated is not None,
            "action": action,
            "path": path,
            "rule": replacement,
            "replaced": target,
            "job": job_title,
            "reason": why,
        }

    # ------------------------------------------------------- job wiring
    def link_job(self, job_note_path: str, job_title: str) -> bool:
        """Give a Job note its `## Preferences` reference.

        The reference is what loads the preference note at execution time,
        so a Job without one has preferences that can never apply.
        """
        note = self.vault.read(job_note_path)
        if note is None:
            return False
        link = f"- [[{job_preference_title(job_title)}]]"
        existing = extract_section(note.body, JOB_PREFERENCES_SECTION)
        if job_preference_title(job_title) in existing:
            return False

        def mutate(target: Note) -> Note:
            target.body = replace_section(target.body, JOB_PREFERENCES_SECTION, link)
            return target

        self.vault.update_note(job_note_path, mutate)
        self.index.invalidate()
        self.index.refresh()
        return True

    def referenced_note(self, job_note: Note) -> Note | None:
        """The preference note a Job note points at, if any.

        Resolved from the Job's own `## Preferences` wikilink -- an
        explicit reference, which is exactly why it still works while
        preference notes are excluded from scanning.
        """
        for title in extract_list_items(job_note.section(JOB_PREFERENCES_SECTION)):
            summary = self.index.find_by_title(title)
            if summary is not None:
                return self.vault.read(summary.relative_path)
        # A Job whose link is missing still gets its preferences, by the
        # naming convention. The link is for humans and for Obsidian's
        # graph; the convention is the fallback that cannot rot.
        path = job_preference_path(job_note.title)
        return self.vault.read(path) if self.vault.note_exists(path) else None


_STORE: PreferenceStore | None = None


def get_preferences(vault: VaultManager | None = None, index: VaultIndex | None = None) -> PreferenceStore:
    global _STORE
    if vault is not None or index is not None:
        return PreferenceStore(vault=vault, index=index)
    if _STORE is None:
        _STORE = PreferenceStore()
    return _STORE


def reset_preferences() -> None:
    global _STORE
    _STORE = None
