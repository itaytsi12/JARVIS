"""Consolidation: keeping the vault from becoming a pile of contradictions.

The failure this module prevents:

    Stored:  "Keep responses short."
    User:    "When we're coding, I want detailed technical explanations."
    Naive:   both stored, as two absolutes, forever contradicting.

What should happen instead is a single refined rule:

    "Default to concise responses. Coding and debugging work may use a
     detailed technical explanation when it helps."

So a new rule is not appended blindly. It is compared against what is
already there, and one of four things happens:

- **duplicate**   -- already said. Nothing changes.
- **refine**      -- the new rule SCOPES the old one ("when coding..."):
                     the two become one sentence with a default and an
                     exception.
- **supersede**   -- the new rule directly contradicts the old one with no
                     scoping: the old rule moves to `## Superseded` with
                     the date, and the new one takes its place.
- **add**         -- unrelated. Appended.

Provenance is never thrown away. A superseded rule is kept, dated, in the
note itself, because "why does JARVIS behave like this now" has to be
answerable from the vault -- and because a rule the user reverses twice is
information.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from vault.manager import VaultManager
from vault.note import Note, extract_section, replace_section, utc_now

log = logging.getLogger("jarvis.vault.consolidation")

DUPLICATE = "duplicate"
REFINE = "refine"
SUPERSEDE = "supersede"
ADD = "add"

SUPERSEDED_HEADING = "Superseded"

#: Pairs whose presence on opposite sides of two rules about the same
#: subject means they genuinely conflict. Deliberately small and concrete:
#: a general-purpose contradiction detector would be wrong far more often
#: than it was right, and a WRONG supersede silently deletes a real rule.
_OPPOSITES = (
    ({"short", "shorter", "brief", "concise", "terse"}, {"detailed", "detail", "long", "longer", "thorough", "verbose", "comprehensive"}),
    ({"always"}, {"never"}),
    ({"open", "launch", "start"}, {"reuse", "existing", "focus"}),
    ({"ask", "confirm"}, {"without", "skip"}),
    ({"enable", "on"}, {"disable", "off"}),
)

#: A clause that limits WHEN a rule applies. Its presence is what turns a
#: contradiction into a refinement.
_SCOPE = re.compile(
    r"\b(when|whenever|while|during|for|if|unless|in|on)\b\s+(?!the\s+(?:user|answer)\b)[\w\s-]{2,40}?"
    r"\b(work|working|task|tasks|mode|code|coding|debug|debugging|test|testing|"
    r"report|reports|question|questions|music|browser|meeting|morning|evening|night)\b",
    re.I,
)

#: Note what is NOT here: "always" and "never". They are ordinary noise in
#: a retrieval query and the entire signal in a rule -- "always confirm
#: before deleting" versus "never confirm before deleting" differ in
#: exactly one word, and stopping it made the two look unrelated.
_STOP = frozenset(
    "a an the and or but to of in on at for with by is are be do does keep make use i you it that this "
    "when whenever while during if unless my your me want prefer should must".split()
)


@dataclass
class ConsolidationResult:
    applied: bool
    action: str
    rule: str = ""
    replaced: str = ""
    reason: str = ""
    note: Note | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "action": self.action,
            "rule": self.rule,
            "replaced": self.replaced,
            "reason": self.reason,
        }


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+", (text or "").lower()) if word not in _STOP and len(word) > 2}


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _contradicts(existing: str, incoming: str) -> bool:
    """Do these two rules take opposite positions on the same subject?

    An opposite PAIR is itself the evidence that the two rules are about
    the same subject; shared vocabulary is not required and demanding it
    was wrong. "Keep responses short." and "When coding, give detailed
    technical explanations." share not one word -- `short` versus
    `detailed` IS the disagreement -- so an overlap test rejected the
    single most important case this module exists to handle.

    The pairs in `_OPPOSITES` are deliberately few and concrete for the
    same reason: a general-purpose contradiction detector would be wrong
    far more often than right, and a wrong "supersede" silently deletes a
    rule the user meant.
    """
    left, right = _tokens(existing), _tokens(incoming)
    for first, second in _OPPOSITES:
        if (left & first and right & second) or (left & second and right & first):
            return True
    return False


def _scope_of(text: str) -> str:
    match = _SCOPE.search(text or "")
    return match.group(0).strip() if match else ""


def classify_against(existing_rules: Iterable[str], incoming: str) -> tuple[str, str, str]:
    """Decide what to do with `incoming`. Returns (action, target, reason)."""
    incoming = (incoming or "").strip()
    if not incoming:
        return ADD, "", "nothing to add"
    for rule in existing_rules:
        rule = rule.strip()
        if not rule:
            continue
        if rule.lower() == incoming.lower() or _overlap(rule, incoming) >= 0.85:
            return DUPLICATE, rule, "the same rule is already recorded"
        if _contradicts(rule, incoming):
            scope = _scope_of(incoming)
            if scope:
                return REFINE, rule, f"the new rule limits the old one to a situation ({scope})"
            return SUPERSEDE, rule, "the new rule contradicts the old one and states no situation that would limit it"
    return ADD, "", "no existing rule covers this"


def merge_scoped(existing: str, incoming: str) -> str:
    """One sentence carrying the default and the scoped exception."""
    default = existing.rstrip(". ").strip()
    exception = incoming.rstrip(". ").strip()
    if default.lower().startswith("default to "):
        return f"{default}; {exception[0].lower()}{exception[1:]}."
    return f"Default: {default[0].lower()}{default[1:]}; {exception[0].lower()}{exception[1:]}."


def _bullets(section_text: str) -> list[str]:
    return [
        line.strip()[2:].strip()
        for line in (section_text or "").splitlines()
        if line.strip().startswith(("- ", "* "))
    ]


def integrate_rule(vault: VaultManager, relative_path: str, section: str, rule: str) -> ConsolidationResult:
    """Add `rule` to one section, consolidating against what is there."""
    note = vault.read(relative_path)
    if note is None:
        return ConsolidationResult(applied=False, action=ADD, rule=rule, reason=f"{relative_path} does not exist")

    body = extract_section(note.body, section)
    existing = _bullets(body)
    numbered = re.findall(r"^\s*\d+\.\s+(.*)$", body, re.M)
    action, target, reason = classify_against([*existing, *numbered], rule)

    if action == DUPLICATE:
        return ConsolidationResult(applied=False, action=DUPLICATE, rule=rule, replaced=target, reason=reason, note=note)

    if action == REFINE:
        merged = merge_scoped(target, rule)
        updated = _replace_line(vault, relative_path, section, target, merged)
        log.info("Refined a rule in %s: %r + %r -> %r", relative_path, target, rule, merged)
        return ConsolidationResult(applied=True, action=REFINE, rule=merged, replaced=target, reason=reason, note=updated)

    if action == SUPERSEDE:
        updated = _replace_line(vault, relative_path, section, target, rule)
        if updated is not None:
            updated = _record_superseded(vault, relative_path, target, rule)
        log.info("Superseded a rule in %s: %r -> %r", relative_path, target, rule)
        return ConsolidationResult(applied=True, action=SUPERSEDE, rule=rule, replaced=target, reason=reason, note=updated)

    def mutate(target_note: Note) -> Note:
        current = extract_section(target_note.body, section)
        entry = f"- {rule}"
        if not current.strip() or current.strip().startswith("_Nothing recorded"):
            merged = entry
        elif numbered and not existing:
            # A numbered procedure stays numbered: appending a bullet to a
            # list of numbered steps produces a note nobody can follow.
            merged = f"{current.rstrip()}\n{len(numbered) + 1}. {rule}"
        else:
            merged = f"{current.rstrip()}\n{entry}"
        target_note.body = replace_section(target_note.body, section, merged)
        return target_note

    updated = vault.update_note(relative_path, mutate)
    return ConsolidationResult(applied=updated is not None, action=ADD, rule=rule, reason=reason, note=updated)


def _replace_line(vault: VaultManager, relative_path: str, section: str, old: str, new: str) -> Note | None:
    def mutate(note: Note) -> Note:
        current = extract_section(note.body, section)
        lines = current.splitlines()
        for position, line in enumerate(lines):
            if old.strip() and old.strip() in line:
                prefix = re.match(r"^(\s*(?:[-*]\s+|\d+\.\s+)?)", line).group(1)
                lines[position] = f"{prefix}{new}"
                break
        else:
            lines.append(f"- {new}")
        note.body = replace_section(note.body, section, "\n".join(lines))
        return note

    return vault.update_note(relative_path, mutate)


def _record_superseded(vault: VaultManager, relative_path: str, old: str, new: str) -> Note | None:
    entry = f"- `{utc_now()[:10]}` ~~{old}~~ -> {new}"

    def mutate(note: Note) -> Note:
        current = extract_section(note.body, SUPERSEDED_HEADING)
        if current.strip().startswith("_Nothing"):
            current = ""
        merged = f"{current.rstrip()}\n{entry}".strip() if current.strip() else entry
        note.body = replace_section(note.body, SUPERSEDED_HEADING, merged)
        return note

    return vault.update_note(relative_path, mutate)


def find_contradictions(vault: VaultManager, relative_path: str, section: str) -> list[tuple[str, str]]:
    """Every pair of rules in one section that take opposite positions.

    A diagnostic, not an automatic fixer. Consolidating a pair the user
    wrote by hand is their decision, not JARVIS's -- this reports them.
    """
    note = vault.read(relative_path)
    if note is None:
        return []
    rules = _bullets(extract_section(note.body, section))
    found: list[tuple[str, str]] = []
    for index, first in enumerate(rules):
        for second in rules[index + 1:]:
            if _contradicts(first, second):
                found.append((first, second))
    return found
