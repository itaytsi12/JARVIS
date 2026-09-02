"""Is this request mission-shaped, or is it a one-shot command?

"Volume down" must stay instant. It does not need a Job, a Skill, a
mission record or a vault scan, and paying for one would make JARVIS
worse at the thing it does most often. "Inspect this project, fix the
error, run it and verify everything" is the opposite: skipping the vault
there is what makes JARVIS rediscover the same facts every week.

So there is one decision, made once, offline, with no model call, before
any vault work happens at all. It is deliberately conservative in BOTH
directions:

- A request that a deterministic route already handled never reaches this
  module at all -- `brain/agent_service.py::run_agent_task` is only called
  for requests that already needed real reasoning.
- Within those, a trivial one ("what time is it", "open notepad") is
  primed cheaply (identity and preferences only), not fully.

The vault is never skipped entirely for an agent task: even a trivial one
gets the core rules and the user's preferences, which are small and are
what make JARVIS behave like JARVIS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Full mission treatment: a Job, Skills, a persistent mission record,
#: knowledge updates at the end.
FULL = "full"
#: Light priming: identity and preferences only, no mission record.
LIGHT = "light"

#: Verbs that mean the request CHANGES something or has to be worked out.
_SUBSTANTIAL = re.compile(
    r"\b(fix|repair|debug|diagnose|implement|build|create|write|refactor|rewrite|patch|"
    r"migrate|optimi[sz]e|troubleshoot|investigate|analyse|analyze|review|audit|"
    r"research|plan|design|organi[sz]e|clean\s+up|set\s+up|configure|install|deploy|"
    r"run\s+the\s+tests?|verify|test|generate|produce|summari[sz]e|compare|"
    r"figure\s+out|work\s+out|root\s+cause|why\s+(?:is|does|did|are|isn'?t|doesn'?t))\b",
    re.I,
)

#: Signs of a request with several steps or a real deliverable.
_MULTI_STEP = re.compile(
    r"\b(then|after that|and then|next,|first|finally|step\s+\d|"
    r"and (?:also |then )?(?:make|run|check|verify|fix|write|open))\b",
    re.I,
)

#: A request naming a long-running or overnight job.
_LONG_RUNNING = re.compile(r"\b(overnight|tonight|while i(?:'m| am) (?:asleep|away|out)|in the background|keep (?:going|running))\b", re.I)


@dataclass(frozen=True)
class MissionPolicy:
    mode: str
    reasons: tuple[str, ...] = ()
    #: Should a persistent mission note be created?
    persist_mission: bool = False
    #: How many characters of vault knowledge this request may load.
    budget_chars: int = 1500

    @property
    def is_full(self) -> bool:
        return self.mode == FULL

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "persist_mission": self.persist_mission,
            "budget_chars": self.budget_chars,
            "why": "; ".join(self.reasons),
        }


def assess(request: str, *, budget_chars: int = 6000) -> MissionPolicy:
    """Decide how much vault work this request deserves. No model call."""
    text = (request or "").strip()
    if not text:
        return MissionPolicy(mode=LIGHT, reasons=("empty request",))

    reasons: list[str] = []
    score = 0

    if _SUBSTANTIAL.search(text):
        score += 2
        reasons.append("names work that changes something or must be reasoned out")
    if _MULTI_STEP.search(text):
        score += 2
        reasons.append("has several steps")
    if _LONG_RUNNING.search(text):
        score += 3
        reasons.append("long-running or unattended")
    if len(text.split()) >= 12:
        score += 1
        reasons.append("a long request")
    if text.count(".") + text.count(",") >= 2 and len(text.split()) >= 8:
        score += 1
        reasons.append("several clauses")

    if score >= 2:
        return MissionPolicy(mode=FULL, reasons=tuple(reasons), persist_mission=True, budget_chars=budget_chars)
    return MissionPolicy(
        mode=LIGHT,
        reasons=tuple(reasons or ("short, single-step request",)),
        persist_mission=False,
        budget_chars=min(1500, budget_chars),
    )
