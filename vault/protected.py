"""Protected knowledge: what a conversation is not allowed to change.

JARVIS may improve its own Jobs, Skills, Lessons, preferences, project
notes, daily notes and mission state. It may not weaken a safety rule
because of something said in passing. "Don't ask me about deleting things
every time" is a real, understandable request, and it must not silently
become "never confirm a destructive action again".

Two independent guards, both required:

1. **A protected NOTE is never edited automatically.** Anything under
   `system/`, plus any note whose frontmatter carries `protected: true`.
   The user edits those in Obsidian, by hand, or not at all.
2. **A protected TOPIC is never weakened, wherever it appears.** A
   correction that would remove a confirmation step, disable a safety
   check, or authorise a destructive or outward-facing action is refused
   even when the note it targets is an ordinary Skill.

A refusal is never silent: it becomes a recorded, visible request for the
user to make the change themselves.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from vault.note import Note

log = logging.getLogger("jarvis.vault.protected")

#: Everything under these directories is protected.
PROTECTED_DIRECTORIES = ("system/",)

#: Topics no automatic edit may weaken, wherever they are written.
_PROTECTED_TOPICS = re.compile(
    r"\b(confirm(?:ation|ing)?|approval|permission|safety|safeguard|guard\s?rail|"
    r"credential|password|api[\s_-]?key|token|secret|"
    r"payment|purchase|billing|invoice|transfer\s+money|"
    r"delete|deleting|erase|wipe|format|uninstall|rm\s+-rf|"
    r"irreversible|destructive)\b",
    re.I,
)

#: Phrasings that would REMOVE a safeguard rather than change a behaviour.
_WEAKENING = re.compile(
    r"\b(stop\s+asking|don'?t\s+ask|no\s+need\s+to\s+(?:ask|confirm|check)|"
    r"without\s+(?:asking|confirming|approval|permission)|"
    r"skip\s+(?:the\s+)?(?:confirmation|check|approval|safety)|"
    r"disable|turn\s+off|ignore|bypass|override|never\s+confirm|"
    r"just\s+do\s+it\s+(?:without|instead\s+of)|stop\s+checking)\b",
    re.I,
)


@dataclass(frozen=True)
class ProtectionVerdict:
    """Whether an automatic edit may proceed, and why not if it may not."""

    allowed: bool
    reason: str = ""
    #: What the user would have to do instead. Empty when allowed.
    manual_action: str = ""

    def describe(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "manual_action": self.manual_action}


def is_protected_path(relative_path: str) -> bool:
    path = (relative_path or "").replace("\\", "/").lstrip("./")
    return any(path.startswith(directory) for directory in PROTECTED_DIRECTORIES)


def is_protected_note(note: Note | None) -> bool:
    if note is None:
        return False
    if is_protected_path(note.relative_path):
        return True
    flag = note.metadata.get("protected")
    if isinstance(flag, bool):
        return flag
    return str(flag or "").strip().lower() in {"true", "yes", "1"}


def weakens_a_safeguard(text: str) -> bool:
    """Does this correction try to REMOVE a safety behaviour?

    Both halves must be present: a protected topic AND weakening language.
    "Always confirm before deleting" names a protected topic and is a
    STRENGTHENING, so it is allowed. "Stop asking me before you open
    Notepad" is weakening language on a topic that is not protected, so it
    is also allowed. Only the intersection is refused.
    """
    body = text or ""
    return bool(_PROTECTED_TOPICS.search(body) and _WEAKENING.search(body))


def check_edit(note: Note | None, *, correction: str = "", relative_path: str = "") -> ProtectionVerdict:
    """May JARVIS apply this automatic edit?

    Called by every automatic write path -- the learning engine, the
    consolidation pass, and the agent-facing `vault_update_note` tool.
    """
    path = relative_path or (note.relative_path if note else "")
    if is_protected_note(note) or is_protected_path(path):
        return ProtectionVerdict(
            allowed=False,
            reason=f"{path or 'that note'} is protected knowledge and is never edited automatically.",
            manual_action=f"Open {path or 'the note'} in Obsidian and change it by hand.",
        )
    if correction and weakens_a_safeguard(correction):
        return ProtectionVerdict(
            allowed=False,
            reason="That correction would weaken a safety rule (confirmation, credentials, money, or a destructive action).",
            manual_action="Edit system/protected_rules.md in Obsidian if this really should change.",
        )
    return ProtectionVerdict(allowed=True)
