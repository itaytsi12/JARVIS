"""One mission's whole vault lifecycle, in one object.

`brain/agent_service.py::run_agent_task` calls exactly three methods:

    session = VaultSession.begin(goal)      # scan, prime, create the mission
    session.observe_step(...)               # as tools run
    session.finish(...)                     # verify, learn, record, complete

Everything the milestones describe -- Job selection, Skill loading,
persistent mission records, successful-method learning, project memory,
the Daily Note -- happens inside those three calls, so the agent path
gains all of it without growing a second orchestration layer of its own.

The hard rule here is that **the vault can never break a request.** Every
public method is wrapped: a vault failure is logged and the mission
continues without it. Memory that occasionally fails to record is a
degraded feature; an assistant that refuses to work because a note could
not be written is a broken one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.daily import DailyJournal, get_journal
from vault.index import VaultIndex, get_index
from vault.learning import CorrectionLearner, LearningOutcome, classify_feedback, get_correction_learner
from vault.manager import VaultManager, get_vault
from vault.missions import Mission, MissionStore, get_mission_store
from vault.policy import FULL, MissionPolicy, assess
from vault.priming import PrimedContext, Primer, get_primer
from vault.projects import ProjectMemory, get_project_memory
from vault.skills import SkillLibrary, get_skill_library

log = logging.getLogger("jarvis.vault.session")


def _publish(event: str, **payload: Any) -> None:
    """Tell whoever is watching what stage this mission is at.

    Goes through `config/events.py`, the one bus every layer already uses,
    so the vault never imports `voice/*` or `ui/*` -- the same
    one-directional rule `brain/activity_state.py` follows. With nothing
    subscribed (the CLI, the tests, `--no-ui`) this is a dict lookup on an
    empty list.
    """
    try:
        from config import events

        events.publish(event, **payload)
    except Exception:  # pragma: no cover - a UI hook must never break a mission
        log.debug("Could not publish %s", event, exc_info=True)


@dataclass
class StepObservation:
    """One tool call, as the mission record cares about it."""

    tool: str
    success: bool
    detail: str = ""
    error: str = ""


@dataclass
class VaultSession:
    """The vault side of one agent task."""

    goal: str
    policy: MissionPolicy
    primed: PrimedContext | None = None
    mission: Mission | None = None
    vault: VaultManager = field(default_factory=get_vault)
    index: VaultIndex | None = None
    journal: DailyJournal | None = None
    projects: ProjectMemory | None = None
    skills: SkillLibrary | None = None
    steps: list[StepObservation] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    enabled: bool = True

    # ------------------------------------------------------------ begin
    @classmethod
    def begin(
        cls,
        goal: str,
        *,
        vault: VaultManager | None = None,
        index: VaultIndex | None = None,
        primer: Primer | None = None,
        missions: MissionStore | None = None,
        journal: DailyJournal | None = None,
        budget_chars: int = 6000,
        task_id: str | None = None,
        enabled: bool = True,
    ) -> "VaultSession":
        """Scan, prime and (for a real mission) create the mission note."""
        policy = assess(goal, budget_chars=budget_chars)
        vault = vault or get_vault()
        session = cls(goal=goal, policy=policy, vault=vault, enabled=enabled)
        if not enabled:
            return session
        try:
            session.index = index or get_index(vault)
            session.journal = journal or get_journal(vault=vault, index=session.index)
            session.projects = get_project_memory(vault=vault, index=session.index)
            session.skills = get_skill_library(index=session.index, vault=vault)

            from vault.bootstrap import ensure_vault_ready

            ensure_vault_ready(vault)

            primer = primer or get_primer(vault=vault, index=session.index)
            _publish("vault.scanning", detail="Scanning long-term memory")
            session.primed = primer.prime(
                goal,
                budget_chars=policy.budget_chars,
                include_continuity=policy.is_full,
            )
            log.info("Vault priming for %r:\n%s", goal[:80], session.primed.explain())
            _publish(
                "vault.reading",
                detail=f"Reading {len(session.primed.notes_read)} of {session.primed.scanned} notes",
                scanned=session.primed.scanned,
                read=len(session.primed.notes_read),
                job=session.primed.job_title,
            )

            if policy.persist_mission:
                store = missions or get_mission_store(vault=vault, index=session.index)
                session.mission = store.create(
                    goal,
                    job=session.primed.job_title,
                    skills=session.primed.skill_titles,
                    project=session.primed.project.title if session.primed.project else "",
                    task_id=task_id,
                )
                session.mission.record_knowledge(
                    job=session.primed.job_title,
                    skills=session.primed.skill_titles,
                    notes=session.primed.notes_read,
                    rationale=f"scanned {session.primed.scanned} note summaries; read {len(session.primed.notes_read)} in full",
                )
        except Exception:
            log.exception("Vault priming failed; the request continues without vault knowledge")
        return session

    # ------------------------------------------------------------ during
    def extra_context(self) -> dict[str, str]:
        """The named blocks for `ContextBuilder.build(extra=...)`."""
        return self.primed.extra_sections() if self.primed is not None else {}

    def observe_step(self, tool: str, *, success: bool, detail: str = "", error: str = "") -> None:
        """Record one tool call. Cheap, and appended to disk as it happens."""
        observation = StepObservation(tool=tool, success=success, detail=detail, error=error)
        self.steps.append(observation)
        if not success and error:
            self.failures.append(f"{tool}: {error}")
        if self.mission is None:
            return
        try:
            if success:
                self.mission.append_progress(f"`{tool}` succeeded." + (f" {detail}" if detail else ""), step=tool)
            else:
                self.mission.append_failure(f"`{tool}` failed: {error or 'no error text'}")
        except Exception:  # pragma: no cover - a note failure must not stop a tool
            log.exception("Could not record a step on mission %s", self.mission.mission_id)

    def note_discovery(self, text: str) -> None:
        if self.mission is None or not text.strip():
            return
        try:
            self.mission.append_discovery(text)
        except Exception:  # pragma: no cover
            log.exception("Could not record a discovery")

    # ------------------------------------------------------------ finish
    def finish(
        self,
        *,
        success: bool,
        verified: bool,
        answer: str,
        errors: Iterable[str] = (),
        artifacts: Iterable[str] = (),
        stop_reason: str = "",
    ) -> dict[str, Any]:
        """Complete the mission, learn what is durable, and record the day."""
        report: dict[str, Any] = {"mission": None, "learned": [], "daily": False}
        if not self.enabled:
            return report
        try:
            error_list = [str(item) for item in errors if str(item).strip()]
            _publish("vault.learning", detail="Recording what this mission learned")
            learned = self._learn_successful_method(success=success, verified=verified)
            report["learned"] = learned

            if self.mission is not None:
                for artifact in artifacts:
                    self.mission.append_artifact(str(artifact))
                self.mission.complete(success=success, outcome=answer or stop_reason or "(no answer recorded)", verified=verified)
                report["mission"] = self.mission.relative_path

            self._record_daily(success=success, verified=verified, answer=answer, errors=error_list, artifacts=artifacts, learned=learned)
            report["daily"] = True

            self._update_project(success=success, verified=verified)
        except Exception:
            log.exception("Recording the mission outcome in the vault failed")
        return report

    def _learn_successful_method(self, *, success: bool, verified: bool) -> list[str]:
        """Milestone 12: never pay the discovery tax twice.

        Only recorded when the mission genuinely succeeded AND something
        failed on the way -- that combination is exactly what "method A
        failed, method B worked" looks like, and it is the only case where
        the sequence is worth writing down. A mission that succeeded on
        the first try discovered nothing.
        """
        if not (success and verified) or not self.failures or self.skills is None or self.primed is None:
            return []
        if not self.primed.skills:
            return []
        successful = [step.tool for step in self.steps if step.success]
        if not successful:
            return []
        failed_tools = sorted({step.tool for step in self.steps if not step.success})
        method = f"For this kind of work, `{successful[-1]}` produced the verified result."
        failures = [f"`{tool}` failed here first" for tool in failed_tools]

        learned: list[str] = []
        target = self.primed.skills[0]
        updated = self.skills.record_working_method(
            target.relative_path,
            method=method,
            failed_attempts=failures,
            source=f"mission {self.mission.mission_id}" if self.mission else "an agent run",
        )
        if updated is not None:
            learned.append(target.title)
            log.info("Recorded a working method on Skill %s", target.title)
        return learned

    def _record_daily(
        self,
        *,
        success: bool,
        verified: bool,
        answer: str,
        errors: list[str],
        artifacts: Iterable[str],
        learned: list[str],
    ) -> None:
        if self.journal is None:
            return
        today = self.journal.today()
        outcome = "Succeeded" if success else "Did not succeed"
        if success and verified:
            outcome = "Succeeded (verified)"
        did_parts = []
        if self.primed is not None:
            if self.primed.job_title:
                did_parts.append(f"Used the [[{self.primed.job_title}]] Job.")
            if self.primed.skill_titles:
                did_parts.append("Loaded " + ", ".join(f"[[{title}]]" for title in self.primed.skill_titles) + ".")
        tools = [step.tool for step in self.steps]
        if tools:
            did_parts.append(f"Ran {len(tools)} tool calls: {', '.join(sorted(set(tools))[:8])}.")

        today.add_event(
            _headline(self.goal),
            request=self.goal,
            did=" ".join(did_parts),
            result=f"{outcome}. {answer.strip()[:400]}",
            files=list(artifacts)[:6],
            lesson="; ".join(learned) if learned else "",
        )
        for error in errors[:3]:
            today.add_problem(error[:300])
        for title in learned:
            today.add_method(f"Recorded a working method on [[{title}]].")
        if self.primed is not None and self.primed.project is not None:
            today.add_project_update(self.primed.project.title)
        if not success:
            today.add_unfinished(f"{_headline(self.goal)} -- {answer.strip()[:200] or 'did not complete'}")
            today.add_next_action(f"Retry: {_headline(self.goal)}")
        today.refresh_quick_summary()

    def _update_project(self, *, success: bool, verified: bool) -> None:
        """Milestone 14, applied with restraint.

        Only a verified success updates the project's recent work, and
        only a failure records an unresolved task. Nothing else is
        permanent -- a project note that grew on every request would stop
        being worth reading, which is the failure mode this guards.
        """
        if self.projects is None or self.primed is None or self.primed.project is None:
            return
        title = self.primed.project.title
        if success and verified:
            self.projects.record_recent_work(title, _headline(self.goal))
        elif not success:
            self.projects.record_unresolved(title, f"{_headline(self.goal)} did not complete.")

    # ---------------------------------------------------------- learning
    def apply_correction(self, correction: str, *, learner: CorrectionLearner | None = None) -> LearningOutcome:
        """Milestone 8: turn user feedback into durable knowledge.

        The notes this mission was PRIMED on are offered as the candidates,
        because they are the notes that actually governed the behaviour
        being corrected.
        """
        feedback = classify_feedback(correction)
        if feedback.kind != "persistent":
            return LearningOutcome(applied=False, kind=feedback.kind, reason="This applies to the current task only.")
        try:
            learner = learner or get_correction_learner(vault=self.vault, index=self.index)
            candidates = list(self.primed.notes_read) if self.primed is not None else []
            outcome = learner.apply(correction, candidate_paths=candidates, feedback=feedback)
            if outcome.applied and self.journal is not None:
                self.journal.today().add_correction(
                    f"{outcome.rule} (recorded in [[{outcome.target_title}]], section '{outcome.section}')"
                )
                self.journal.today().refresh_quick_summary()
            elif outcome.protection is not None and self.journal is not None:
                self.journal.today().add_correction(
                    f"REFUSED an automatic change: {outcome.reason} {outcome.manual_action if outcome.protection else ''}".strip()
                )
            return outcome
        except Exception:
            log.exception("Applying a correction to the vault failed")
            return LearningOutcome(applied=False, kind=feedback.kind, reason="The vault could not be updated.")

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.policy.describe(),
            "priming": self.primed.describe() if self.primed else None,
            "mission": self.mission.describe() if self.mission else None,
            "steps": len(self.steps),
            "failures": len(self.failures),
        }


def _headline(text: str) -> str:
    import re

    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned if len(cleaned) <= 80 else cleaned[:77].rsplit(" ", 1)[0] + "..."
