"""Resolve short conversational follow-ups against real recent state.

`brain/router.py` is a cascade of deterministic patterns matched purely on
the current utterance's TEXT. That is correct for a self-contained command
("open Spotify"), but a follow-up like "What does that mean?" or "Batman
instead." is not self-contained -- its meaning depends on what JARVIS just
did or said. Before this module existed, `route_command` had no access to
session state at all, so a generic explanatory follow-up fell through to
`brain/request_intent.py`'s QUESTION classification and was answered by
`brain/web_answer.py` -- a context-blind web search for a question that was
never about the web.

Two independent resolvers live here, one per confirmed live bug:

- `resolve_explanatory_followup` -- "what does that mean", "why", "explain
  that", ... resolved against `SessionContext.last_assistant_response`
  (already the single field every route's final answer/result/error text
  funnels through -- see `brain/agent.py::run_agent`).
- `resolve_browser_search_correction` -- "Batman instead.", "try Superman
  instead.", "change that to X." resolved against
  `SessionContext.browser_active` / `last_search_query` /
  `last_search_provider`, reusing the exact search-URL templates the
  deterministic router and local planner already use for "search youtube
  for X" -- so a correction to an existing browser search stays a single
  local `browser_open_url` action and never needs a model call.

Both resolvers return `None` (never raise, never guess) when the pattern
does not match or the referenced context does not exist -- callers fall
through to the pre-existing routing exactly as before. Neither is a special
case for the example sentences in the bug report: the patterns describe the
general phrasing class (an explanatory question word, or a short "X
instead"/"change it to X" correction), not any one wording.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

from brain.models import Action
from brain.request_complexity import looks_like_simple_target

# -- explanatory follow-ups -------------------------------------------------

#: Whole-phrase matches. Kept as a fixed set (not a loose keyword regex) so
#: an ordinary command that happens to contain "why" or "explain" is never
#: hijacked -- these are exactly the short, context-only follow-ups named in
#: the bug report, plus their natural variants.
_EXPLANATORY_PHRASES = {
    "what does that mean",
    "what does this mean",
    "what does it mean",
    "what did that mean",
    "why",
    "why is that",
    "why did that happen",
    "why did that happen?",
    "why is that happening",
    "why not",
    "explain that",
    "explain this",
    "explain",
    "what happened",
    "what just happened",
    "tell me more",
    "tell me more about that",
    "tell me more about this",
    "can you explain that",
    "can you explain",
}

_EXPLANATORY_RE = re.compile(
    r"^(?:why|what does (?:that|this|it) mean|explain (?:that|this)?|what happened|what just happened|tell me more(?: about (?:that|this))?)\??$",
    re.I,
)


def is_explanatory_followup(text: str) -> bool:
    """Is `text` a generic "explain the last thing" follow-up?

    Deliberately narrow: only phrasings that carry no content of their own
    (no named subject, no verb naming a new action) qualify. A command that
    happens to start with "why" but names something concrete
    ("why is Chrome using so much memory") is NOT here -- it has its own
    subject and is a normal question, not a reference to prior context.
    """
    normalized = " ".join((text or "").lower().strip().rstrip(".!").split())
    if not normalized:
        return False
    if normalized in _EXPLANATORY_PHRASES:
        return True
    return bool(_EXPLANATORY_RE.fullmatch(normalized))


def resolve_recent_referent(context) -> str | None:
    """The most relevant recent assistant output to explain, or None.

    `last_assistant_response` is set unconditionally at the end of every
    `run_agent` call (success, tool error, or answered question alike), so
    it already IS "the most relevant recent assistant answer / task result
    / error / command result" -- there is no separate field to prefer over
    it.
    """
    if context is None:
        return None
    referent = getattr(context, "last_assistant_response", None) or getattr(context, "last_spoken_response", None)
    return referent or None


@dataclass(frozen=True)
class ContextualQuestionRoute:
    message: str
    context_text: str

    def as_dict(self) -> dict:
        return {
            "type": "contextual_question",
            "message": self.message,
            "context_text": self.context_text,
            "route_source": "conversational_context",
        }


def resolve_explanatory_followup(command: str, context) -> dict | None:
    """`{"type": "contextual_question", ...}` when `command` is a generic
    explanatory follow-up AND a real recent referent exists, else `None` --
    in which case normal question handling continues unchanged."""
    if not is_explanatory_followup(command):
        return None
    referent = resolve_recent_referent(context)
    if not referent:
        return None
    return ContextualQuestionRoute(message=command.strip(), context_text=referent).as_dict()


# -- browser search corrections ---------------------------------------------

#: The same provider -> search-URL templates `brain/router.py` and
#: `brain/local_planner.py` already use for "search youtube for X" /
#: "search google for X". Centralized here instead of a third copy so a
#: correction always lands on the exact same URL shape the original search
#: would have used.
SEARCH_URL_TEMPLATES: dict[str, str] = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
    "reddit": "https://www.reddit.com/search/?q={}",
    "github": "https://github.com/search?q={}",
}

_INSTEAD_RE = re.compile(r"^(?:search for|search|try)\s+(?P<query>.+?)\s+instead$", re.I)
_BARE_INSTEAD_RE = re.compile(r"^(?P<query>.+?)\s+instead$", re.I)
_CHANGE_TO_RE = re.compile(r"^change (?:that|it) to\s+(?P<query>.+)$", re.I)


def _extract_correction_query(text: str) -> str | None:
    stripped = text.strip().rstrip(".!?")
    for pattern in (_INSTEAD_RE, _CHANGE_TO_RE, _BARE_INSTEAD_RE):
        match = pattern.match(stripped)
        if match:
            query = match.group("query").strip()
            if query:
                return query
    return None


def resolve_browser_search_correction(text: str, context) -> dict | None:
    """`{"type": "local_plan", ...}` re-running the last browser search with
    a corrected query, or `None` when the phrase does not describe an
    obvious search-query replacement, or the current session has no
    resolvable browser search to correct -- in which case the caller falls
    through to its normal routing (including, eventually, the agent
    runtime, which can still resolve a genuinely ambiguous correction like
    "search for something else")."""
    if context is None or not getattr(context, "browser_active", False):
        return None
    # "send it to Alex instead" is a WhatsApp-recipient correction
    # (brain/router.py's dedicated `revise_whatsapp_recipient` pattern), not
    # a browser search -- excluded explicitly so a coincidentally-active
    # browser session can never hijack it.
    if re.match(r"^(?:don't\s+)?send\b", text.strip(), re.I):
        return None
    provider = getattr(context, "last_search_provider", None)
    template = SEARCH_URL_TEMPLATES.get(provider or "")
    if template is None:
        return None
    query = _extract_correction_query(text)
    if not query or not looks_like_simple_target(query):
        return None
    url = template.format(urllib.parse.quote_plus(query))
    return {
        "type": "local_plan",
        "actions": [Action(tool="browser_open_url", args={"url": url})],
        "route_source": "conversational_context_browser_correction",
    }


__all__ = [
    "SEARCH_URL_TEMPLATES",
    "is_explanatory_followup",
    "resolve_recent_referent",
    "resolve_explanatory_followup",
    "resolve_browser_search_correction",
]
