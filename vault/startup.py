"""Startup memory recovery: what JARVIS knows before the first request.

A new session must not begin blank. At startup JARVIS reads, cheaply:

- its identity and core rules,
- the user's preferences,
- the active project,
- any mission left running when the last process stopped,
- today's Daily Note and the most recent previous one,
- the current-state note.

Summaries first, in every case. This is deliberately NOT a deep read of
everything: it is a few hundred characters that let a first request like
"continue what we were doing yesterday" or "open the project I was working
on" resolve against something real. The deep read happens later, per
mission, in `vault/priming.py`.

It is also where an interrupted mission is recognised. Any mission still
sitting in `missions/active/` when a new process starts is, by
definition, one whose owner is gone -- so it is marked `interrupted`
rather than left claiming to be running. Marking it is honest; guessing
its outcome would not be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vault.bootstrap import ensure_vault_ready
from vault.daily import DailyJournal, get_journal
from vault.index import VaultIndex, get_index
from vault.manager import VaultManager, get_vault
from vault.missions import Mission, MissionStore, get_mission_store
from vault.note import Note

log = logging.getLogger("jarvis.vault.startup")


@dataclass
class SessionRecovery:
    """What a fresh session recovered from the vault."""

    vault_root: str = ""
    notes: int = 0
    identity_summary: str = ""
    preferences: str = ""
    active_project: str = ""
    current_state: str = ""
    today: str = ""
    previous_day: str = ""
    resumable_missions: list[dict[str, Any]] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)
    scan_ms: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "vault_root": self.vault_root,
            "notes": self.notes,
            "active_project": self.active_project,
            "resumable_missions": [item.get("mission_id") for item in self.resumable_missions],
            "interrupted": list(self.interrupted),
            "has_today": bool(self.today),
            "has_previous_day": bool(self.previous_day),
            "scan_ms": round(self.scan_ms, 2),
        }

    def context_text(self, *, max_chars: int = 2000) -> str:
        """The recovered context, as one block for a session's first turn."""
        parts: list[str] = []
        if self.preferences:
            parts.append(f"## What the user prefers\n{self.preferences}")
        if self.active_project:
            parts.append(f"## Active project\n{self.active_project}")
        if self.current_state:
            parts.append(f"## Where things stand\n{self.current_state}")
        if self.resumable_missions:
            lines = [
                f"- {item.get('title') or item.get('mission_id')} ({item.get('status')}): next step {item.get('current_step') or 'unknown'}"
                for item in self.resumable_missions
            ]
            parts.append("## Missions left unfinished\n" + "\n".join(lines))
        if self.today:
            parts.append(self.today)
        if self.previous_day:
            parts.append(self.previous_day)
        return "\n\n".join(parts)[:max_chars]

    def spoken_summary(self) -> str:
        """One short, honest sentence for the user, if there is anything to say."""
        if self.resumable_missions:
            first = self.resumable_missions[0]
            return f"There is an unfinished mission from earlier, sir: {first.get('title') or first.get('mission_id')}."
        return ""


def recover_session(
    vault: VaultManager | None = None,
    index: VaultIndex | None = None,
    journal: DailyJournal | None = None,
    missions: MissionStore | None = None,
    *,
    mark_interrupted: bool = True,
    create_today: bool = True,
) -> SessionRecovery:
    """Read the small things a new session needs. Never raises.

    A vault problem must never stop JARVIS from starting: every failure
    here is logged and produces an empty recovery, and the assistant comes
    up exactly as it would have without a vault.
    """
    import time

    started = time.perf_counter()
    recovery = SessionRecovery()
    try:
        vault = ensure_vault_ready(vault or get_vault())
        index = index or get_index(vault)
        journal = journal or get_journal(vault=vault, index=index)
        missions = missions or get_mission_store(vault=vault, index=index)

        recovery.vault_root = str(vault.root)
        summaries = index.refresh()
        recovery.notes = len(summaries)

        identity = vault.read("identity/jarvis.md")
        if identity is not None:
            recovery.identity_summary = identity.summary

        preferences = vault.read("user/preferences.md")
        if preferences is not None:
            recovery.preferences = (preferences.section("Preferences") or preferences.quick_summary)[:600]

        state = vault.read("state/current.md")
        if state is not None:
            recovery.active_project = state.section("Active Project").strip()[:200]
            unfinished = state.section("Unfinished Work").strip()
            recovery.current_state = unfinished[:400] if unfinished and not unfinished.startswith("_Nothing") else ""

        if mark_interrupted:
            for mission in missions.mark_orphans_interrupted():
                recovery.interrupted.append(mission.relative_path)
        recovery.resumable_missions = [mission.describe() for mission in missions.resumable()[:5]]

        if create_today:
            today = journal.today()
            recovery.today = today.brief(max_chars=700)
        else:
            existing = journal.existing(journal.today().date)
            recovery.today = existing.brief(max_chars=700) if existing else ""

        previous = journal.yesterday()
        if previous is not None:
            recovery.previous_day = previous.brief(max_chars=700)

    except Exception:
        log.exception("Vault startup recovery failed; continuing without it")
    recovery.scan_ms = (time.perf_counter() - started) * 1000
    log.info("Vault startup recovery: %s", recovery.describe())
    return recovery


def record_session_start(vault: VaultManager | None = None, journal: DailyJournal | None = None, *, detail: str = "") -> None:
    """Note in today's Daily Note that a session began. Never raises."""
    try:
        vault = vault or get_vault()
        journal = journal or get_journal(vault=vault)
        journal.today().add_event("JARVIS session started", did=detail or "Started up and recovered context from the vault.")
    except Exception:  # pragma: no cover - a journal failure must not stop startup
        log.exception("Could not record the session start in the daily note")
