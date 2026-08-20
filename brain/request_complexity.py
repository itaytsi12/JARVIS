"""Whole-request complexity assessment for the deterministic router.

`brain/router.py` is a cascade of deterministic patterns, several of which
end in an open-ended capture (`inspect (.+)`, `switch to (.+)`,
`press (.+)`, ...). A pattern like that matches on ONE keyword and then
swallows the entire remainder of the sentence as an argument, so a request
such as

    "Inspect this JARVIS project and tell me how the main components
     are connected. Do not modify anything."

used to resolve to `inspect_window(app_name="this jarvis project and tell
me how ...")` -- a single local UI-accessibility call that cannot possibly
satisfy the request, and which then failed.

This module supplies the two general guards that fix that class of bug,
rather than any single phrasing:

- `looks_like_simple_target` -- COVERAGE. A free-form capture is only a
  valid tool argument when what it captured actually looks like a target
  (an app or window name), not like the rest of a sentence. This is what
  keeps a keyword from forcing a single local tool.
- `assess_complexity` -- SCOPE. A whole-request assessment answering the
  question the router never asked: can ANY single deterministic action
  satisfy this entire request? When it cannot, the request belongs to the
  agent runtime.

Both are pure, offline and deterministic -- no LLM call and no network,
in the same family as `brain/request_intent.py` and
`brain/coding_task_intent.py`. The design rule shared with those modules
holds here too: a bare keyword is never sufficient evidence on its own.

Deliberately NOT implemented as a special case for the word "project" or
for the sentence above. The escalation rules below combine independent
signals (reasoning required, operation count, domain reference, local-
machine reference, scoping constraint), so a phrasing that never mentions
a project still escalates when it is genuinely agentic, and a short
command that does mention one still stays on the local fast path.

`requires_local_context` additionally answers a question the QUESTION
route could not: "is this answerable from the web at all?" A question
about a file on disk, a directory, or `git status` is not, so
`brain/router.py` sends it to the agent runtime (which has the
filesystem/terminal tools) rather than to `brain/web_answer.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Clause separators. A request that chains several operations is a
# sequence, and a single tool call cannot be the whole of it.
_OPERATION_SPLIT_RE = re.compile(
    r"\s*(?:,|;|\band then\b|\bthen\b|\band also\b|\balso\b|\bafter that\b|\bafter which\b|\band\b)\s*",
    re.I,
)

# Any verb that makes a clause an actual operation rather than a fragment.
_VERB_RE = re.compile(
    r"\b(open|launch|start|run|execute|close|quit|inspect|check|read|look|search|find|fix|"
    r"debug|repair|refactor|implement|patch|test|verify|explain|describe|tell|show|list|"
    r"analyz|analys|review|investigate|diagnose|trace|compare|summariz|understand|figure|"
    r"write|edit|change|modify|update|add|remove|delete|click|type|press|switch|focus|play|"
    r"pause|save|download|install|build|deploy|connect)\w*\b",
    re.I,
)

# Reasoning / open-ended result. These ask for an ANSWER that has to be
# worked out -- something no deterministic tool in `tools/` can return,
# because the result is not a state change. Deliberately restricted to the
# how/why/analysis forms: a bare "tell me" is NOT here, so an ordinary
# factual question ("tell me what the weather is") still reaches the
# existing web-answer question route instead of the agent.
_REASONING_RE = re.compile(
    r"("
    r"\b(?:tell|show|explain|teach|walk)\s+me\s+(?:about\s+)?(?:how|why|what.{0,20}\bdo(?:es)?\b)\b"
    r"|\bexplain\b(?!\s+that\s+to)"
    r"|\b(?:figure|work)\s+out\b"
    r"|\bfind\s+out\s+(?:how|why|what)\b"
    r"|\bunderstand\s+(?:how|why|what)\b"
    r"|\b(?:analyz|analys|investigat|diagnos|troubleshoot)\w*\b"
    r"|\breview\s+(?:the|this|my|our)\b"
    r"|\bsummariz\w*\b"
    r"|\bwalk\s+through\b"
    r"|\bhow\s+(?:it|they|this|that|these|those|the\s+\w+)\s+\w*\s*(?:work|works|connect|connects|"
    r"fit|fits|interact|interacts|relate|relates|is|are)\b"
    r"|\bwhy\s+(?:it|they|this|that|the\s+\w+)\s+\w*\s*(?:fail|fails|failing|break|breaks|broken|"
    r"crash|crashes|error|errors|is|are|does|doesn't|not)\b"
    r"|\bfind\s+(?:the|any|all)\s+(?:bug|bugs|issue|issues|problem|problems|error|errors|cause)\b"
    r"|\bwhat(?:'s| is)\s+(?:wrong|going on|happening|causing)\b"
    r"|\broot\s+cause\b"
    r"|\bhow\s+(?:do|does|did)\s+(?:it|they|this|the)\b"
    r")",
    re.I,
)

# Software-domain reference: a project/repository/codebase/source. One
# signal among several -- never sufficient on its own (see the rules in
# `assess_complexity`), so "open vs code" or "save the file" stay local.
_CODEBASE_RE = re.compile(
    r"\b(project|repository|repositories|repo|repos|codebase|code\s?base|source\s+code|"
    r"module|modules|package|packages|class|classes|function|functions|"
    r"test\s+suite|unit\s+tests|the\s+tests|architecture|component|components|"
    r"implementation|stack\s?trace|traceback)\b",
    re.I,
)
# Weaker domain words: real evidence, but ambiguous enough on their own
# ("open vs code", "save the file") that they only count alongside a
# stronger signal.
_WEAK_CODEBASE_RE = re.compile(r"\b(code|script|scripts|file|files|folder|directory|bug|bugs)\b", re.I)

# LOCAL CONTEXT: the request is about THIS machine -- a file on disk, a
# folder, this project, a shell/git command. This is the signal that
# separates "answerable from the web" from "answerable only by looking",
# and it is what allows a QUESTION to reach the agent runtime (which has
# filesystem/terminal/code tools) instead of the web-answer service, which
# by construction cannot see any of it.
#
# Each alternative below requires a concrete referent -- a real filename
# with an extension, a possessive/demonstrative pointing at a project or
# directory, or an actual command name -- so an ordinary "open the file
# explorer" or "save the file" never matches.
_LOCAL_CONTEXT_RE = re.compile(
    r"\b[\w\-]+\.(?:py|pyw|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|cfg|ini|env|html|css|sh|bat|ps1|sql|rs|go|java|rb|c|h|cpp)\b"
    r"|\b(?:this|the|my|our)\s+(?:\w+\s+){0,2}?"
    r"(?:project|repo|repository|codebase|code\s?base|source\s+code|folder|directory|working\s+tree)\b"
    r"|\b(?:git|npm|pip|pytest|poetry|cargo|docker|kubectl)\s+[a-z][\w\-]*\b"
    r"|\b(?:in|from|on)\s+(?:the\s+)?(?:terminal|command\s+line|shell|console)\b"
    r"|\bcurrent\s+(?:directory|folder)\b"
    r"|\bworking\s+directory\b",
    re.I,
)

# A scoping constraint ("do not modify anything") describes HOW a task
# should be carried out. A single deterministic tool call has no way to
# honour an instruction like this, so its presence is evidence the request
# is a task rather than a command.
_CONSTRAINT_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|without|never)\s+(?:modify|change|edit|touch|write|delete|break|alter)\b"
    r"|\bread[- ]only\b|\bdry[- ]run\b",
    re.I,
)

# Conjunctions / sentence structure that a genuine single-target argument
# never contains. Used by the coverage guard.
_TARGET_STOPWORD_RE = re.compile(
    r"\b(and|then|also|but|or|so\s+that|because|while|after|before|explain|tell|show|find|"
    r"figure|why|how|what|which|whether|please\s+also)\b",
    re.I,
)

# Verbs that READ local state (as opposed to acting on an app). Paired
# with `_LOCAL_CONTEXT_RE` in `assess_complexity`: "list the files in this
# project", "read main.py", "run git status and show me the output".
_INSPECTION_RE = re.compile(
    r"\b(?:list|show|tell|read|inspect|look|check|find|search|see|display|print|"
    r"describe|explain|summariz\w*|analyz\w*|analys\w*|review|diff|run|execute|"
    r"what|which|where|how\s+many)\b",
    re.I,
)

MAX_SIMPLE_TARGET_WORDS = 5


@dataclass(frozen=True)
class ComplexityAssessment:
    """Why a request does or does not exceed a single deterministic action."""

    is_complex: bool
    operations: int
    requires_reasoning: bool
    references_codebase: bool
    has_constraint: bool
    reason: str
    signals: tuple[str, ...] = field(default_factory=tuple)
    #: The request is about THIS machine (a file, a folder, this project, a
    #: shell command) and therefore cannot be answered from the web.
    requires_local_context: bool = False

    def describe(self) -> dict:
        return {
            "is_complex": self.is_complex,
            "operations": self.operations,
            "requires_reasoning": self.requires_reasoning,
            "references_codebase": self.references_codebase,
            "requires_local_context": self.requires_local_context,
            "has_constraint": self.has_constraint,
            "reason": self.reason,
            "signals": list(self.signals),
        }


def count_operations(text: str) -> int:
    """How many distinct operations the request asks for.

    Counts clauses that actually contain a verb, so "open my project,
    inspect the error, fix it and test again" is four operations while
    "open the file explorer" stays one and a trailing fragment
    ("...and quickly") does not inflate the count.
    """
    normalized = (text or "").strip()
    if not normalized:
        return 0
    # Sentence boundaries separate operations too, but a trailing "." must
    # not create an empty extra clause.
    clauses: list[str] = []
    for sentence in re.split(r"[.?!]+", normalized):
        clauses.extend(_OPERATION_SPLIT_RE.split(sentence))
    operations = sum(1 for clause in clauses if clause.strip() and _VERB_RE.search(clause))
    return operations or (1 if _VERB_RE.search(normalized) else 0)


def looks_like_simple_target(captured: str) -> bool:
    """Is `captured` a plausible single target (an app/window name)?

    This is the COVERAGE guard for the router's open-ended capture
    patterns. `inspect (.+)` will happily capture an entire paragraph;
    this is what decides whether what it captured can honestly be used as
    a tool argument. "notepad" and "the active window" pass; "this jarvis
    project and tell me how the main components are connected" does not.
    """
    target = (captured or "").strip().strip("\"'").rstrip(".?!,;:")
    if not target:
        return False
    if len(target.split()) > MAX_SIMPLE_TARGET_WORDS:
        return False
    # Internal sentence punctuation means more than one clause was caught.
    if re.search(r"[.;,!?]", target):
        return False
    if _TARGET_STOPWORD_RE.search(target):
        return False
    return True


def assess_complexity(text: str) -> ComplexityAssessment:
    """Decide whether any single deterministic action could satisfy the
    WHOLE request.

    The rules combine independent signals rather than matching phrasings:

    1. Reasoning is required AND the request is either about software or
       chains several operations. An open-ended answer is not something
       `tools/` can produce, so a single tool route cannot be complete.
    2. Three or more chained operations in a software context -- past the
       point where one tool call, or the rule-based local planner's
       fixed action shapes, can represent the request.
    3. A scoping constraint ("do not modify anything") alongside a
       software reference and any analysis: a constraint on HOW to work is
       meaningless to a one-shot tool call.
    4. An INSPECTION of this machine -- a real filename, "this project",
       "git status" -- paired with a reading verb. `tools/` has no route
       for these and the web-answer service cannot see the disk, so no
       single deterministic action can serve one however it is phrased.
       Reported as `local_inspection_requires_tools`, which
       `brain/router.py` treats differently from the other three: it names
       work ONLY the agent runtime can do, so with no agent configured the
       request keeps its pre-existing route instead.

    Everything else is NOT complex, which keeps the deterministic fast
    path -- "open Spotify", "volume down", "inspect window" -- untouched.
    Note the deliberate asymmetry in rule 4: "open my downloads folder"
    names a local referent but its verb ACTS rather than reads, so it stays
    on `open_path`.
    """
    normalized = " ".join((text or "").lower().strip().split())
    if not normalized:
        return ComplexityAssessment(False, 0, False, False, False, "empty")

    operations = count_operations(normalized)
    requires_reasoning = bool(_REASONING_RE.search(normalized))
    local_context = bool(_LOCAL_CONTEXT_RE.search(normalized))
    # A concrete local referent (a real filename, "this project", "git
    # status") is at least as strong a software-domain signal as the
    # generic vocabulary in `_CODEBASE_RE`, and much more specific.
    strong_domain = bool(_CODEBASE_RE.search(normalized)) or local_context
    weak_domain = bool(_WEAK_CODEBASE_RE.search(normalized))
    references_codebase = strong_domain or weak_domain
    has_constraint = bool(_CONSTRAINT_RE.search(normalized))

    signals: list[str] = []
    if requires_reasoning:
        signals.append("requires_reasoning")
    if local_context:
        signals.append("requires_local_context")
    if strong_domain:
        signals.append("references_codebase")
    elif weak_domain:
        signals.append("references_code_weak")
    if operations >= 2:
        signals.append(f"operations={operations}")
    if has_constraint:
        signals.append("scoping_constraint")

    def _result(is_complex: bool, reason: str) -> ComplexityAssessment:
        return ComplexityAssessment(
            is_complex, operations, requires_reasoning, references_codebase,
            has_constraint, reason, tuple(signals), local_context,
        )

    if requires_reasoning and (references_codebase or operations >= 2):
        return _result(True, "open_ended_reasoning_beyond_single_action")
    if operations >= 3 and strong_domain:
        return _result(True, "multi_operation_software_task")
    if has_constraint and strong_domain and operations >= 2:
        return _result(True, "constrained_software_task")
    # An INSPECTION of this machine: reading a real file, listing a real
    # directory, running a real command. `tools/` has no route for these at
    # all and the web-answer service cannot see the disk, so a single
    # deterministic action can never satisfy one -- regardless of how the
    # sentence is phrased or how many clauses it has.
    if local_context and _INSPECTION_RE.search(normalized):
        return _result(True, "local_inspection_requires_tools")
    return _result(False, "single_action_sufficient")


def exceeds_single_action(text: str) -> bool:
    """Convenience predicate: `assess_complexity(text).is_complex`."""
    return assess_complexity(text).is_complex
