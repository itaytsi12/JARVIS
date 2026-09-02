"""Learning from corrections: turning "no, do it this way" into knowledge.

The user says:

    "No. When Apple Music is already open, don't open another one.
     Use the existing window."

That is not an instruction for the current task. It is a rule, and next
week's session has to already know it. So JARVIS:

1. classifies the feedback as PERSISTENT rather than one-time,
2. finds the note that actually governed the behaviour,
3. rewrites the relevant rule cleanly -- not by pasting the sentence,
4. updates that note's Quick Summary if the change is material,
5. refreshes `updated`,
6. records the change in today's Daily Note,
7. and uses the corrected rule from the next session onwards.

## One-time versus persistent

    "Make this answer shorter."          -> one-time. Nothing is written.
    "From now on keep reports shorter."  -> persistent. A preference is written.

The distinction is made semantically, not by an exact-phrase list. Strong
scope markers ("from now on", "always", "never", "next time", "remember
that", "whenever I ask") raise it; equally strong immediacy markers
("this time", "just for now", "in this case", "for this one") lower it,
and they WIN when both appear -- "just make this one shorter, always
happy to read the long version otherwise" must not become a standing rule.

## What is deliberately NOT done

The user's sentence is never pasted verbatim into a note as a rule. It is
converted into an imperative statement in the note's own voice, because a
Procedure section made of quoted complaints is not something a future
session can follow.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from vault.consolidation import integrate_rule
from vault.index import VaultIndex, get_index
from vault.manager import VaultManager
from vault.note import JOB, SKILL, USER, Note, extract_section, replace_section, utc_now
from vault.protected import ProtectionVerdict, check_edit
from vault.retrieval import VaultRetriever, get_retriever

log = logging.getLogger("jarvis.vault.learning")

#: How well a note must match before an automatic edit will TARGET it.
#: Higher than the retrieval threshold on purpose -- see `find_target`.
_EDIT_TARGET_MIN_SCORE = 4.0

ONE_TIME = "one_time"
PERSISTENT = "persistent"
NOT_A_CORRECTION = "not_a_correction"

#: Scope markers that make feedback durable.
_PERSISTENT_MARKERS = (
    (r"\bfrom now on\b", 3.0),
    (r"\bgoing forward\b", 3.0),
    (r"\bin (?:the )?future\b", 2.5),
    (r"\balways\b", 2.5),
    (r"\bnever\b", 2.5),
    (r"\bnext time\b", 2.5),
    (r"\bevery time\b", 2.5),
    (r"\bwhenever (?:i|you|we)\b", 2.5),
    (r"\bdon'?t do that again\b", 3.0),
    (r"\bstop doing that\b", 2.0),
    (r"\bremember (?:that|to|this)\b", 3.0),
    (r"\bby default\b", 2.0),
    (r"\bas a rule\b", 2.0),
    (r"\bthis is how i want\b", 3.0),
    (r"\bi prefer\b", 2.0),
    (r"\bi (?:always|never) want\b", 3.0),
    (r"\bmake (?:it|that) (?:the|your) default\b", 3.0),
)

#: Immediacy markers that keep feedback local to the current task. These
#: outrank the persistent markers when both are present.
_ONE_TIME_MARKERS = (
    (r"\bthis time\b", 3.0),
    (r"\bjust (?:this|for) (?:once|now|time)\b", 3.0),
    (r"\bfor (?:this|the current) (?:one|task|answer|case|report)\b", 3.0),
    (r"\bin this case\b", 2.5),
    (r"\bfor now\b", 2.5),
    (r"\bright now\b", 1.5),
    (r"\bthis (?:answer|reply|message|response|one)\b", 2.0),
)

#: Feedback at all -- a correction has to be correcting SOMETHING.
_CORRECTION_MARKERS = (
    r"\bno[,.\s]", r"^no$", r"\bnot like that\b", r"\bwrong\b", r"\bincorrect\b",
    r"\bdon'?t\b", r"\bdo not\b", r"\bstop\b", r"\binstead\b", r"\bshould(?:n'?t)? (?:have|be)\b",
    r"\bi (?:said|asked|wanted|meant)\b", r"\bthat'?s not\b", r"\brather than\b",
    r"\bprefer\b", r"\bremember\b", r"\balways\b", r"\bnever\b", r"\bfrom now on\b",
    r"\bnext time\b", r"\buse the\b",
    # An imperative aimed at JARVIS's own output. "Make this answer
    # shorter" is unmistakably feedback and matched none of the patterns
    # above, so it was classified as "not a correction at all" -- which
    # skipped the one-time/persistent decision entirely and meant a
    # genuinely durable version of the same sentence could never be
    # learned either.
    r"\b(?:make|keep|give|write|say|answer|reply)\b[^.?!]{0,40}"
    r"\b(?:shorter|longer|briefer|terser|concise|detailed|simpler|clearer|verbose)\b",
    r"\b(?:make|keep)\s+(?:it|this|that|them|these|the)\b",
)


@dataclass
class Feedback:
    """One classified piece of user feedback."""

    text: str
    kind: str
    confidence: float
    signals: list[str] = field(default_factory=list)

    @property
    def is_persistent(self) -> bool:
        return self.kind == PERSISTENT

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "confidence": round(self.confidence, 2), "signals": list(self.signals), "text": self.text[:200]}


def classify_feedback(text: str) -> Feedback:
    """Is this feedback, and if so does it change anything permanently?

    Deterministic, offline, no model call -- it runs on every user
    utterance, so it has to be free.
    """
    body = (text or "").strip()
    if not body:
        return Feedback(text=body, kind=NOT_A_CORRECTION, confidence=0.0)

    lowered = body.lower()
    signals: list[str] = []
    if not any(re.search(pattern, lowered, re.M) for pattern in _CORRECTION_MARKERS):
        return Feedback(text=body, kind=NOT_A_CORRECTION, confidence=0.0)

    persistent_score = 0.0
    for pattern, weight in _PERSISTENT_MARKERS:
        if re.search(pattern, lowered):
            persistent_score += weight
            signals.append(f"persistent: {pattern}")

    one_time_score = 0.0
    for pattern, weight in _ONE_TIME_MARKERS:
        if re.search(pattern, lowered):
            one_time_score += weight
            signals.append(f"one-time: {pattern}")

    # A conditional clause -- "when X is already open, do Y" -- states the
    # SITUATION a rule applies in, which is what makes a rule reusable
    # rather than a one-off. It is a genuine persistence signal even with
    # no explicit "from now on".
    if re.search(r"\b(?:when|whenever|if)\b[^.?!]{4,80}\b(?:,|then)\s", lowered) or re.search(
        r"\b(?:when|whenever|if) (?:it'?s|it is|there'?s|there is|.{0,30}\bis\b) already\b", lowered
    ):
        persistent_score += 2.0
        signals.append("persistent: states the situation a rule applies in")

    if one_time_score >= persistent_score and one_time_score > 0:
        # Immediacy wins ties on purpose: wrongly writing a standing rule
        # is far more damaging than wrongly treating one request as local.
        return Feedback(text=body, kind=ONE_TIME, confidence=min(1.0, one_time_score / 3.0), signals=signals)
    if persistent_score >= 2.0:
        return Feedback(text=body, kind=PERSISTENT, confidence=min(1.0, persistent_score / 3.0), signals=signals)
    return Feedback(text=body, kind=ONE_TIME, confidence=0.4, signals=signals or ["no scope marker; treated as this task only"])


# --------------------------------------------------------------- rewriting

_IMPERATIVE_PREFIX = re.compile(
    r"^\s*(?:no[,.\s]+|nope[,.\s]+|wrong[,.\s]+|that'?s (?:not right|wrong)[,.\s]+|"
    r"actually[,.\s]+|please[,.\s]+|jarvis[,.\s]+|hey jarvis[,.\s]+|"
    r"from now on[,.\s]+|going forward[,.\s]+|in future[,.\s]+|in the future[,.\s]+|"
    r"next time[,.\s]+|remember (?:that|to)[,.\s]*|i (?:want|need) you to[,.\s]*|"
    r"you should[,.\s]*|you need to[,.\s]*|i'?d (?:like|prefer) (?:you )?to[,.\s]*)+",
    re.I,
)


def rewrite_as_rule(text: str) -> str:
    """Turn a spoken correction into a clean, imperative rule.

    Deliberately conservative and purely mechanical: it strips the
    conversational scaffolding, normalises the person, and returns a
    single imperative sentence. It never invents content the user did not
    say -- an "improved" rule that says something the user did not is far
    worse than a slightly clumsy one that says exactly what they did.
    """
    body = (text or "").strip()
    if not body:
        return ""
    body = _IMPERATIVE_PREFIX.sub("", body).strip()
    body = re.sub(r"^(?:you\s+)?(?:should|must|need to|have to)\s+", "", body, flags=re.I)
    body = re.sub(r"\bi want you to\b\s*", "", body, flags=re.I)
    body = re.sub(r"\bplease\b\s*", "", body, flags=re.I)
    body = re.sub(r"\s+", " ", body).strip(" ,;")
    if not body:
        return ""
    body = body[0].upper() + body[1:]
    if not body.endswith((".", "!", "?")):
        body += "."
    return body


# ------------------------------------------------------------- application


@dataclass
class LearningOutcome:
    """What the correction actually changed. Never a claim, always a record."""

    applied: bool
    kind: str
    target_path: str = ""
    target_title: str = ""
    rule: str = ""
    section: str = ""
    reason: str = ""
    protection: ProtectionVerdict | None = None
    summary_updated: bool = False

    def describe(self) -> dict[str, Any]:
        payload = {
            "applied": self.applied,
            "kind": self.kind,
            "target": self.target_path,
            "rule": self.rule,
            "section": self.section,
            "reason": self.reason,
            "summary_updated": self.summary_updated,
        }
        if self.protection is not None:
            payload["protection"] = self.protection.describe()
        return payload


class CorrectionLearner:
    """Applies a persistent correction to the note that governs the behaviour."""

    def __init__(self, vault: VaultManager | None = None, index: VaultIndex | None = None, retriever: VaultRetriever | None = None):
        self.index = index or get_index(vault)
        self.vault = vault or self.index.vault
        self.retriever = retriever or get_retriever(index=self.index, vault=self.vault)

    def find_target(
        self,
        correction: str,
        *,
        candidate_paths: Iterable[str] = (),
        preferred_types: Iterable[str] = (SKILL, JOB),
    ) -> Note | None:
        """Which note taught the behaviour being corrected?

        `candidate_paths` -- the notes the mission actually primed on -- is
        checked FIRST and preferred, because those are the notes that
        genuinely governed what JARVIS just did. Falling back to a fresh
        scan of the whole vault is the second-best answer, used when the
        correction arrives outside a primed mission.
        """
        candidates = [path for path in candidate_paths if self.vault.note_exists(path)]
        if candidates:
            scored, _, _ = self.retriever.scan(correction)
            by_path = {item.relative_path: item.score for item in scored}
            best = max(candidates, key=lambda path: by_path.get(path, 0.0))
            note = self.vault.read(best)
            # A primed note only wins if it is a KIND of note that carries
            # behaviour. A project or mission note primed alongside a Skill
            # is context, not the rule that was wrong.
            if note is not None and note.note_type in set(preferred_types):
                return note
            for path in candidates:
                other = self.vault.read(path)
                if other is not None and other.note_type in set(preferred_types):
                    return other

        # The fallback scan, used when the correction arrives outside a
        # primed mission. The bar is deliberately far higher than for
        # ordinary retrieval: reading a marginally relevant note wastes a
        # little context, but EDITING one writes a rule into a note that
        # never governed the behaviour. A single weak field match once
        # sent "keep your spoken reports shorter" into the Clipping Job,
        # because that note happens to mention a morning report.
        scored, _, _ = self.retriever.scan(correction, types=list(preferred_types))
        for candidate in scored:
            if candidate.score >= _EDIT_TARGET_MIN_SCORE:
                return self.vault.read(candidate.relative_path)
        return None

    def apply(
        self,
        correction: str,
        *,
        candidate_paths: Iterable[str] = (),
        feedback: Feedback | None = None,
        target_path: str = "",
    ) -> LearningOutcome:
        """Classify, locate, rewrite and record. The whole Milestone 8 loop."""
        feedback = feedback or classify_feedback(correction)
        if feedback.kind != PERSISTENT:
            return LearningOutcome(
                applied=False,
                kind=feedback.kind,
                reason="Feedback applies to the current task only; nothing was written to long-term memory.",
            )

        rule = rewrite_as_rule(correction)
        if not rule:
            return LearningOutcome(applied=False, kind=feedback.kind, reason="Nothing usable remained after removing the conversational wrapping.")

        note = self.vault.read(target_path) if target_path else self.find_target(correction, candidate_paths=candidate_paths)
        if note is None:
            # No Job or Skill governs it, so it is a statement about how
            # the USER wants JARVIS to behave. Preferences are the correct
            # home for that, and they are consolidated rather than piled up.
            return self._apply_to_preferences(rule, feedback)

        verdict = check_edit(note, correction=correction)
        if not verdict.allowed:
            log.info("Refused an automatic edit to %s: %s", note.relative_path, verdict.reason)
            return LearningOutcome(
                applied=False,
                kind=feedback.kind,
                target_path=note.relative_path,
                target_title=note.title,
                rule=rule,
                reason=verdict.reason,
                protection=verdict,
            )

        section = _section_for(note, correction)
        updated = self._write_rule(note, section=section, rule=rule)
        summary_updated = self._maybe_update_summary(updated or note, rule)
        self.index.invalidate()
        self.index.refresh()
        log.info("Correction applied to %s (%s): %s", note.relative_path, section, rule)
        return LearningOutcome(
            applied=True,
            kind=feedback.kind,
            target_path=note.relative_path,
            target_title=note.title,
            rule=rule,
            section=section,
            summary_updated=summary_updated,
            reason=f"Recorded in the '{section}' section of {note.title}.",
        )

    def _apply_to_preferences(self, rule: str, feedback: Feedback) -> LearningOutcome:
        path = "user/preferences.md"
        note = self.vault.read(path)
        if note is None:
            return LearningOutcome(applied=False, kind=feedback.kind, rule=rule, reason="No preferences note exists to record this in.")
        result = integrate_rule(self.vault, path, "Preferences", rule)
        self.index.invalidate()
        self.index.refresh()
        return LearningOutcome(
            applied=result.applied,
            kind=feedback.kind,
            target_path=path,
            target_title=note.title,
            rule=result.rule or rule,
            section="Preferences",
            reason=result.reason,
        )

    def _write_rule(self, note: Note, *, section: str, rule: str) -> Note | None:
        """Integrate the rule into one section, replacing what it supersedes."""
        result = integrate_rule(self.vault, note.relative_path, section, rule)
        return result.note

    def _maybe_update_summary(self, note: Note, rule: str) -> bool:
        """Refresh the Quick Summary when the change is material.

        Material means the rule changed the PROCEDURE -- how the work is
        done -- rather than adding a note to a list of known problems. A
        summary that is rewritten on every correction stops being a
        summary.
        """
        if not note or note.section("Procedure") is None:
            return False
        quick = note.quick_summary
        if not quick or rule.strip() in quick:
            return False
        first_clause = rule.split(".")[0].strip()
        if not first_clause:
            return False
        merged = f"{quick.rstrip()}\n- {first_clause}."

        def mutate(target: Note) -> Note:
            target.body = replace_section(target.body, "Quick Summary", merged)
            return target

        return self.vault.update_note(note.relative_path, mutate) is not None


def _section_for(note: Note, correction: str) -> str:
    """Which section of the note should carry this rule?

    A correction about HOW to do something belongs in the procedure. One
    that names a failure belongs with the known problems. Anything else is
    a lesson learned, which every Job and Skill note has.
    """
    lowered = (correction or "").lower()
    headings = {name.lower(): name for name in note.sections()}
    if re.search(r"\b(fail|failed|broke|broken|doesn'?t work|does not work|error)\b", lowered):
        for candidate in ("known problems", "known problems / traps"):
            if candidate in headings:
                return headings[candidate]
    for candidate in ("procedure", "known working method"):
        if candidate in headings:
            return headings[candidate]
    if "lessons learned" in headings:
        return headings["lessons learned"]
    return "Lessons Learned"


_LEARNER: CorrectionLearner | None = None


def get_correction_learner(vault: VaultManager | None = None, index: VaultIndex | None = None) -> CorrectionLearner:
    global _LEARNER
    if vault is not None or index is not None:
        return CorrectionLearner(vault=vault, index=index)
    if _LEARNER is None:
        _LEARNER = CorrectionLearner()
    return _LEARNER


def reset_correction_learner() -> None:
    global _LEARNER
    _LEARNER = None
