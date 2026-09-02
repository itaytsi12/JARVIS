"""The vault tools the agent is offered.

Priming hands the model the knowledge a mission needs before it starts.
These tools are for what priming could not anticipate: searching for a
note the request did not name, reading one in full, and -- most
importantly -- WRITING back what was learned, so a discovery made during
a run survives the run.

They return the project's existing `brain/models.py::ToolResult`, and
they are dispatched at the one existing dispatch point
(`brain/tool_router.py::execute_tool`) and described in the one existing
catalog (`brain/tool_catalog.py::DEFINITIONS`). There is no second
executor and no second result type.

Every write goes through `vault/protected.py::check_edit` first, so a
model-authored tool call is held to exactly the same rules as an
automatic correction: it can improve a Job, a Skill or a project note,
and it can never edit `system/` or weaken a safety rule.
"""
from __future__ import annotations

import logging
from typing import Any

from vault.index import get_index
from vault.manager import OutsideVault, VaultError, get_vault
from vault.note import KNOWN_TYPES, OTHER, extract_section, replace_section
from vault.protected import check_edit
from vault.retrieval import get_retriever

log = logging.getLogger("jarvis.vault.tools")

#: A deep read has to be bounded, or one tool call can blow the whole
#: context budget the priming layer just spent care staying inside.
MAX_NOTE_CHARS = 6000
MAX_SEARCH_RESULTS = 12
#: How well a note must match before it is reported to the model as a
#: search hit. One incidental word in a quick summary is not a result.
MIN_SEARCH_SCORE = 2.0


def _result(success: bool, tool: str, message: str, data: dict[str, Any] | None = None, error: str | None = None) -> dict:
    """The dict shape `brain/tool_router.py` already returns everywhere."""
    payload = {"success": success, "message": message, "error": error}
    payload.update(data or {})
    return payload


def vault_search(query: str, note_type: str = "", limit: int = 8) -> dict:
    """Stage 1 only: rank note SUMMARIES and return them. Reads no bodies."""
    query = (query or "").strip()
    if not query:
        return _result(False, "vault_search", "I need something to search for.", error="empty_query")
    try:
        retriever = get_retriever()
        types = [note_type.strip().lower()] if note_type.strip() else None
        candidates, scanned, scan_ms = retriever.scan(query, types=types)
        # A floor, because this result goes to the MODEL. `scan` reports
        # everything that scored at all so a trace can explain a decision,
        # but a single incidental word match ("nothing" appearing in a
        # note's quick summary) is noise here, and noise in a tool result
        # is something the model may act on.
        candidates = [item for item in candidates if item.score >= MIN_SEARCH_SCORE]
        top = candidates[: max(1, min(int(limit or 8), MAX_SEARCH_RESULTS))]
        if not top:
            return _result(
                True,
                "vault_search",
                f"Nothing in the vault matches that. I scanned {scanned} note summaries.",
                {"verified": True, "scanned": scanned, "results": []},
            )
        lines = [
            f"- {item.summary.title} ({item.relative_path}) [{item.summary.note_type}] -- {item.summary.summary}"
            for item in top
        ]
        return _result(
            True,
            "vault_search",
            f"Scanned {scanned} note summaries; the closest {len(top)}:\n" + "\n".join(lines),
            {
                "verified": True,
                "scanned": scanned,
                "scan_ms": round(scan_ms, 1),
                "results": [item.describe() for item in top],
                "summary": f"{len(top)} of {scanned} notes matched",
            },
        )
    except Exception as exc:
        log.exception("vault_search failed")
        return _result(False, "vault_search", "I could not search the vault.", error=f"{type(exc).__name__}: {exc}")


def vault_read_note(path: str, section: str = "") -> dict:
    """Stage 2: read one note in full, or just one of its sections."""
    if not (path or "").strip():
        return _result(False, "vault_read_note", "I need a note path.", error="empty_path")
    try:
        note = get_vault().read(path)
    except OutsideVault as exc:
        return _result(False, "vault_read_note", "That path is outside the vault.", error=str(exc))
    except VaultError as exc:
        return _result(False, "vault_read_note", "I could not read that note.", error=str(exc))
    if note is None:
        return _result(False, "vault_read_note", f"There is no note at {path}.", error="note_not_found")

    if section.strip():
        text = extract_section(note.body, section.strip())
        if not text:
            return _result(
                False,
                "vault_read_note",
                f"{note.title} has no '{section}' section. It has: {', '.join(note.sections()) or '(none)'}.",
                {"sections": note.sections()},
                error="section_not_found",
            )
    else:
        text = note.body.strip()

    truncated = len(text) > MAX_NOTE_CHARS
    if truncated:
        text = text[:MAX_NOTE_CHARS].rstrip() + "\n\n[truncated -- ask for one section by name to read the rest]"
    return _result(
        True,
        "vault_read_note",
        f"{note.title} ({note.relative_path}):\n\n{text}",
        {
            "verified": True,
            "title": note.title,
            "type": note.note_type,
            "tags": note.tags,
            "sections": note.sections(),
            "truncated": truncated,
            "summary": note.summary,
        },
    )


def vault_write_note(
    path: str,
    title: str,
    note_type: str,
    summary: str,
    content: str,
    tags: str = "",
    quick_summary: str = "",
) -> dict:
    """Create a new knowledge note. The summary is required, by design."""
    if not (summary or "").strip():
        return _result(
            False,
            "vault_write_note",
            "A note needs a one-sentence summary -- it is how the note is found again without being read.",
            error="summary_required",
        )
    kind = (note_type or "").strip().lower() or OTHER
    if kind not in KNOWN_TYPES:
        return _result(
            False,
            "vault_write_note",
            f"'{note_type}' is not a note type. Use one of: {', '.join(sorted(KNOWN_TYPES))}.",
            error="unknown_note_type",
        )
    verdict = check_edit(None, relative_path=path)
    if not verdict.allowed:
        return _result(False, "vault_write_note", verdict.reason, {"manual_action": verdict.manual_action}, error="protected_note")
    try:
        vault = get_vault()
        if vault.note_exists(path):
            return _result(
                False,
                "vault_write_note",
                f"{path} already exists. Use vault_update_note to change it, so nothing is overwritten.",
                error="note_exists",
            )
        tag_list = [item.strip() for item in (tags or "").replace(";", ",").split(",") if item.strip()]
        quick = [line.strip("- ").strip() for line in (quick_summary or "").splitlines() if line.strip()]
        note = vault.create_note(
            path,
            title=title.strip() or path,
            note_type=kind,
            summary=summary.strip(),
            tags=tag_list,
            quick_summary=quick or [summary.strip()],
            sections=[("Detail", content or "")],
        )
        index = get_index()
        index.invalidate()
        index.refresh()
        return _result(
            True,
            "vault_write_note",
            f"Created {note.relative_path}.",
            {"verified": True, "path": note.relative_path, "title": note.title, "summary": note.summary},
        )
    except Exception as exc:
        log.exception("vault_write_note failed")
        return _result(False, "vault_write_note", "I could not create that note.", error=f"{type(exc).__name__}: {exc}")


def vault_update_note(path: str, section: str, content: str, mode: str = "replace") -> dict:
    """Change one section of an existing note.

    Section-scoped on purpose: a whole-note overwrite from a model is how
    a carefully accumulated Skill note loses everything it knew that the
    model did not happen to repeat.
    """
    if not (path or "").strip() or not (section or "").strip():
        return _result(False, "vault_update_note", "I need a note path and a section name.", error="missing_arguments")
    try:
        vault = get_vault()
        note = vault.read(path)
        if note is None:
            return _result(False, "vault_update_note", f"There is no note at {path}.", error="note_not_found")
        verdict = check_edit(note, correction=content)
        if not verdict.allowed:
            return _result(
                False,
                "vault_update_note",
                verdict.reason,
                {"manual_action": verdict.manual_action},
                error="protected_note",
            )

        wanted = (mode or "replace").strip().lower()
        heading = section.strip()

        def mutate(target):
            existing = extract_section(target.body, heading)
            if wanted == "append" and existing and not existing.startswith("_Nothing recorded"):
                merged = f"{existing.rstrip()}\n{content.strip()}"
            else:
                merged = content.strip()
            target.body = replace_section(target.body, heading, merged)
            return target

        updated = vault.update_note(path, mutate)
        index = get_index()
        index.invalidate()
        index.refresh()
        if updated is None:
            return _result(False, "vault_update_note", "The note could not be written.", error="write_failed")
        # Read it back: an edit is not evidence that the edit happened.
        confirmed = extract_section(updated.body, heading)
        verified = bool(content.strip()) and content.strip()[:60] in confirmed
        return _result(
            True,
            "vault_update_note",
            f"Updated the '{heading}' section of {updated.title}.",
            {"verified": verified, "path": updated.relative_path, "section": heading, "summary": updated.summary},
        )
    except Exception as exc:
        log.exception("vault_update_note failed")
        return _result(False, "vault_update_note", "I could not update that note.", error=f"{type(exc).__name__}: {exc}")


def vault_record_lesson(title: str, summary: str, lesson: str, tags: str = "") -> dict:
    """Record something learned by experience as a Lesson note."""
    from vault.paths import LESSONS_DIR

    if not (summary or "").strip() or not (lesson or "").strip():
        return _result(False, "vault_record_lesson", "A lesson needs a summary and the lesson itself.", error="missing_arguments")
    try:
        vault = get_vault()
        path = vault.unique_path(LESSONS_DIR, title or summary, fallback="lesson")
        tag_list = [item.strip() for item in (tags or "").replace(";", ",").split(",") if item.strip()]
        note = vault.create_note(
            path,
            title=(title or summary)[:80],
            note_type="lesson",
            summary=summary.strip(),
            tags=sorted({"lesson", *tag_list}),
            quick_summary=[summary.strip()],
            sections=[("Lesson", lesson.strip()), ("Situation", ""), ("What Works", "")],
        )
        index = get_index()
        index.invalidate()
        index.refresh()
        return _result(
            True,
            "vault_record_lesson",
            f"Recorded the lesson in {note.relative_path}.",
            {"verified": True, "path": note.relative_path, "summary": note.summary},
        )
    except Exception as exc:
        log.exception("vault_record_lesson failed")
        return _result(False, "vault_record_lesson", "I could not record that lesson.", error=f"{type(exc).__name__}: {exc}")


def vault_record_working_method(skill: str, method: str, failed_attempts: str = "") -> dict:
    """Record a method that WORKED on a Skill note, and what did not.

    This is what stops the next session rediscovering the same sequence.
    """
    if not (skill or "").strip() or not (method or "").strip():
        return _result(False, "vault_record_working_method", "I need a Skill name and the method that worked.", error="missing_arguments")
    try:
        from vault.skills import get_skill_library

        library = get_skill_library()
        failures = [item.strip() for item in (failed_attempts or "").replace(";", "\n").splitlines() if item.strip()]
        updated = library.record_working_method(skill.strip(), method=method.strip(), failed_attempts=failures, source="an agent run")
        if updated is None:
            available = ", ".join(library.titles()) or "(none)"
            return _result(
                False,
                "vault_record_working_method",
                f"There is no Skill note called '{skill}'. Skills available: {available}.",
                error="skill_not_found",
            )
        return _result(
            True,
            "vault_record_working_method",
            f"Recorded the working method on {updated.title}.",
            {"verified": True, "path": updated.relative_path, "summary": updated.summary},
        )
    except Exception as exc:
        log.exception("vault_record_working_method failed")
        return _result(False, "vault_record_working_method", "I could not record that method.", error=f"{type(exc).__name__}: {exc}")


def vault_list_jobs() -> dict:
    """Every Job in the vault, with its one-line summary."""
    try:
        from vault.jobs import get_job_registry

        registry = get_job_registry()
        catalog = registry.catalog()
        return _result(
            True,
            "vault_list_jobs",
            catalog or "There are no Job notes in the vault yet.",
            {"verified": True, "jobs": registry.titles(), "summary": f"{len(registry.titles())} jobs"},
        )
    except Exception as exc:
        log.exception("vault_list_jobs failed")
        return _result(False, "vault_list_jobs", "I could not list the Jobs.", error=f"{type(exc).__name__}: {exc}")


def vault_status() -> dict:
    """Where the vault is, how big it is, and what shape it is in."""
    try:
        vault = get_vault()
        index = get_index()
        described = vault.describe()
        described.update(index.statistics())
        malformed = described.get("malformed") or []
        message = (
            f"The vault is at {described['root']} with {described.get('notes', 0)} notes "
            f"({', '.join(f'{count} {name}' for name, count in described.get('by_type', {}).items())})."
        )
        if malformed:
            message += f" {len(malformed)} notes have unreadable frontmatter."
        return _result(True, "vault_status", message, {"verified": True, **described})
    except Exception as exc:
        log.exception("vault_status failed")
        return _result(False, "vault_status", "I could not read the vault's status.", error=f"{type(exc).__name__}: {exc}")
