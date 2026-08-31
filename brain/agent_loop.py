"""The agent loop: goal -> context -> act -> observe -> adapt -> verify.

This is JARVIS's own runtime. A model provider only ever answers ONE turn
at a time (`providers.base.ModelProvider.complete`); everything that makes
the behaviour agentic -- deciding the next action, feeding back the real
observation, retrying, giving up, and deciding whether the goal was
actually met -- happens here, in JARVIS, so it is identical whichever
provider is plugged in and can be tested with no provider at all.

Safety limits (all configurable, none so tight that ordinary multi-step
work fails):

- `max_agent_steps` (default 25): total model turns.
- `max_action_retries` (default 2): repeats of the SAME failing tool call
  with the same arguments; the third identical attempt is refused with an
  explicit observation telling the model to change approach.
- `max_consecutive_failures` (default 4): unbroken run of failing tool
  calls before the loop stops and reports honestly.
- `agent_task_timeout` (default 900s): wall-clock ceiling.

Cancellation is checked before every model call and before every tool
call, so "cancel" takes effect within one step rather than at the end.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from brain.context_builder import BuiltContext, ContextBuilder
from brain.models import ToolResult
from brain.tool_catalog import SESSION_AWARE_CATEGORIES, SESSION_AWARE_TOOLS, ToolCatalog, get_tool_catalog
from config import get_config
from memory.episodic import StepRecord
from providers.base import (
    Message,
    ModelResponse,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Usage,
)
from skills.base import Skill

log = logging.getLogger("jarvis.agent")

# How a run ended. These are the honest outcomes -- there is no "assumed
# success" among them.
COMPLETED = "completed"
NO_PROVIDER = "no_provider"
STEP_LIMIT = "step_limit"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
FAILURE_LIMIT = "failure_limit"
PROVIDER_ERROR = "provider_error"


@dataclass
class AgentLimits:
    max_steps: int
    max_action_retries: int
    max_consecutive_failures: int
    timeout_seconds: float

    @classmethod
    def from_config(cls) -> "AgentLimits":
        config = get_config()
        return cls(
            max_steps=config.max_agent_steps,
            max_action_retries=config.max_action_retries,
            max_consecutive_failures=config.max_consecutive_failures,
            timeout_seconds=config.agent_task_timeout,
        )


@dataclass
class AgentRun:
    """Everything that happened, and how it ended."""

    goal: str
    answer: str = ""
    success: bool = False
    verified: bool = False
    stop_reason: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retries: int = 0
    model_calls: int = 0
    #: How many times independent tool calls were run concurrently instead of
    #: one after another, and how much wall clock that saved (the difference
    #: between the batch's total tool time and its slowest member).
    parallel_batches: int = 0
    parallel_saved_ms: float = 0.0
    #: Milliseconds from the start of the run to the first tool actually
    #: running, and to the first text token of the final answer. None means
    #: it never happened, which is deliberately distinct from zero.
    first_tool_ms: float | None = None
    first_model_event_ms: float | None = None
    effort: str | None = None
    usage: Usage = field(default_factory=lambda: Usage(reported=False))
    estimated_cost_usd: float | None = None
    duration_ms: float = 0.0
    provider: str | None = None
    model: str | None = None
    context: BuiltContext | None = None
    skills: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    def describe(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success": self.success,
            "verified": self.verified,
            "stop_reason": self.stop_reason,
            "steps": len(self.steps),
            "retries": self.retries,
            "parallel_batches": self.parallel_batches,
            "parallel_saved_ms": round(self.parallel_saved_ms, 1),
            "first_tool_ms": round(self.first_tool_ms, 1) if self.first_tool_ms is not None else None,
            "model_first_event_ms": round(self.first_model_event_ms, 1) if self.first_model_event_ms is not None else None,
            "effort": self.effort,
            "errors": len(self.errors),
            # The COUNT alone made a real provider failure undebuggable
            # from the run summary; the message itself is what says
            # whether the key, the model id, or the request was at fault.
            "last_error": self.errors[-1] if self.errors else None,
            "model_calls": self.model_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "duration_ms": round(self.duration_ms, 1),
            "provider": self.provider,
            "model": self.model,
            "skills": list(self.skills),
        }


ProgressCallback = Callable[[str, dict[str, Any]], None]


def _any_refused(calls: list[ToolCall], attempts: dict[str, int], max_retries: int) -> bool:
    """Is any call in this turn already past its retry limit?

    Such a call is refused rather than executed, so the turn is not a clean
    batch and takes the sequential path where that refusal is handled.
    """
    return any(attempts.get(_signature(call), 0) + 1 > max_retries + 1 for call in calls)


class AgentLoop:
    def __init__(
        self,
        provider: Any,
        catalog: ToolCatalog | None = None,
        *,
        limits: AgentLimits | None = None,
        context_builder: ContextBuilder | None = None,
        progress: ProgressCallback | None = None,
        effort: str | None = None,
        on_answer_text: Callable[[str], None] | None = None,
    ):
        self.provider = provider
        self.catalog = catalog or get_tool_catalog()
        self.limits = limits or AgentLimits.from_config()
        self.context_builder = context_builder or ContextBuilder()
        self.progress = progress
        config = get_config()
        self.effort = effort
        self.max_parallel_tools = config.max_parallel_tools
        # When set, PUBLIC assistant text is forwarded as it is generated so
        # the caller can start speaking before the answer is complete. The
        # provider stops forwarding the moment a tool_use block starts, so
        # this never carries a tool payload; internal reasoning is never
        # subscribed to at all.
        self.on_answer_text = on_answer_text

    # ------------------------------------------------------------------
    def run(
        self,
        goal: str,
        *,
        context: BuiltContext,
        skills: Iterable[Skill] = (),
        tool_specs: Iterable[ToolSpec] | None = None,
        cancellation_token: Any = None,
        task: Any = None,
        session_context: Any = None,
    ) -> AgentRun:
        started = time.perf_counter()
        run = AgentRun(goal=goal, context=context, skills=[skill.name for skill in skills])
        run.provider = getattr(self.provider, "name", None)
        run.model = getattr(self.provider, "model", None)
        run.effort = self.effort

        if self.provider is None or not self.provider.is_available():
            run.stop_reason = NO_PROVIDER
            run.answer = (
                "I can't work on that right now, sir. No reasoning model is configured -- "
                "set ANTHROPIC_API_KEY to enable it."
            )
            run.duration_ms = (time.perf_counter() - started) * 1000
            return run

        specs = list(tool_specs) if tool_specs is not None else self.catalog.specs()
        messages: list[Message] = [Message.user(context.user_prompt)]
        attempts: dict[str, int] = {}
        request_id = uuid.uuid4().hex
        checkpoint = None
        successful_actions: dict[str, tuple[str, str]] = {}
        if getattr(self.provider, "name", "") == "multi_model":
            from providers.pool import TaskCheckpoint
            checkpoint = TaskCheckpoint(original_goal=goal)
        consecutive_failures = 0
        deadline = started + self.limits.timeout_seconds

        for step_number in range(self.limits.max_steps):
            if _is_cancelled(cancellation_token):
                return self._stop(run, started, CANCELLED, "I stopped that, sir.")
            if time.perf_counter() > deadline:
                return self._stop(
                    run,
                    started,
                    TIMEOUT,
                    self._partial_answer(run, "I ran out of time on that, sir."),
                )

            try:
                response = self._call_model(messages, context.system_prompt, specs, request_id=request_id, checkpoint=checkpoint)
            except ProviderUnavailable as exc:
                run.errors.append(str(exc))
                log.error("Agent run has no usable provider: %s", exc)
                return self._stop(run, started, NO_PROVIDER, "I can't reach the reasoning model right now, sir.")
            except ProviderError as exc:
                run.errors.append(f"{type(exc).__name__}: {exc}")
                # Logged here as well as in the provider: without it a real
                # 401/404 from the API is indistinguishable from any other
                # failure once it has become the generic PROVIDER_ERROR
                # stop reason.
                log.error(
                    "Agent run failed at the model call: step=%s provider=%s model=%s %s: %s",
                    step_number,
                    run.provider,
                    run.model,
                    type(exc).__name__,
                    exc,
                )
                if isinstance(exc, ProviderRateLimited):
                    message = "The reasoning model is rate limited right now, sir."
                else:
                    message = "The reasoning model failed on that request, sir."
                return self._stop(run, started, PROVIDER_ERROR, self._partial_answer(run, message))

            run.model_calls += 1
            if run.first_model_event_ms is None and response.first_event_ms is not None:
                run.first_model_event_ms = (time.perf_counter() - started) * 1000 - (
                    response.latency_ms - response.first_event_ms
                )
            _accumulate(run, response)
            self._emit("model_turn", {"step": step_number, "text": response.text[:200], "tool_calls": len(response.tool_calls)})

            if not response.tool_calls:
                # The model produced a final answer. It is only reported as
                # a success when nothing in this run is still an unresolved
                # failure -- see `_conclude`.
                return self._conclude(run, started, response)

            messages.append(response.as_message())
            outcomes: list[ToolOutcome] = []
            # Independent read-only calls in one turn are run concurrently
            # (`_parallel_safe`); everything else falls through to the
            # sequential path below completely unchanged. Pre-computing the
            # batch keeps every attempt/retry/step-record rule identical --
            # only the wall clock differs.
            batched: dict[str, ToolResult] = {}
            if self._parallel_safe(response.tool_calls) and not _any_refused(
                response.tool_calls, attempts, self.limits.max_action_retries
            ):
                if run.first_tool_ms is None:
                    run.first_tool_ms = (time.perf_counter() - started) * 1000
                for call in response.tool_calls:
                    self._emit("tool_started", {"tool": call.name, "attempt": 1})
                batched = self._execute_parallel(response.tool_calls, cancellation_token, run)
            for call in response.tool_calls:
                if _is_cancelled(cancellation_token):
                    return self._stop(run, started, CANCELLED, "I stopped that, sir.")

                signature = _signature(call)
                definition = self.catalog.get(call.name)
                prior = successful_actions.get(signature)
                if prior and definition is not None and not definition.read_only and prior[0] != response.provider:
                    outcome_text = "Already completed successfully by the previous model route; this side effect was not executed again. " + prior[1]
                    outcomes.append(ToolOutcome(call.id, outcome_text, is_error=False))
                    self._record_step(run, call, ToolResult(True, call.name, outcome_text, {"deduplicated": True}), outcome_text, 0, response.text)
                    continue
                attempts[signature] = attempts.get(signature, 0) + 1
                attempt = attempts[signature]
                if attempt > self.limits.max_action_retries + 1:
                    outcome_text = (
                        f"Refused: {call.name} has already been tried {attempt - 1} times with these exact "
                        "arguments and failed each time. Change the approach or report that you are blocked."
                    )
                    outcomes.append(ToolOutcome(call.id, outcome_text, is_error=True))
                    run.errors.append(f"{call.name}:repeated_failure")
                    consecutive_failures += 1
                    self._record_step(run, call, None, outcome_text, attempt, response.text)
                    continue

                if attempt > 1:
                    run.retries += 1

                if run.first_tool_ms is None:
                    run.first_tool_ms = (time.perf_counter() - started) * 1000
                # Emitted BEFORE the call so a listener can distinguish "the
                # tool is still running" from "the model is thinking" -- the
                # difference between an honest status line and a guess.
                self._emit("tool_started", {"tool": call.name, "attempt": attempt})
                result = batched.get(call.id) or self.catalog.execute(
                    call.name, call.arguments, cancellation_token=cancellation_token
                )
                if session_context is not None:
                    # Feeds the same structured short-term context a
                    # deterministic plan's tool calls do (section 16) --
                    # e.g. a `run_command` pytest failure list becomes a
                    # `last_result_set` a later "fix the first one" can
                    # resolve against, regardless of which execution path
                    # (deterministic plan vs. agent loop) actually ran it.
                    # Applies to a batched/parallel-prefetched result too:
                    # it is the same ToolResult, just produced earlier in
                    # this same turn.
                    from brain.context_resolver import observe_tool_result

                    observe_tool_result(session_context, call.name, call.arguments, result)
                observation = self.context_builder.bound_observation(_observation_text(result))
                outcomes.append(ToolOutcome(call.id, observation, is_error=not result.success))
                self._record_step(run, call, result, observation, attempt, response.text)
                self._emit(
                    "tool_result",
                    {"tool": call.name, "success": result.success, "error": result.error, "attempt": attempt},
                )
                if result.success:
                    consecutive_failures = 0
                    attempts[signature] = 0
                    successful_actions[signature] = (response.provider, observation[:500])
                    if checkpoint is not None:
                        checkpoint.completed_steps.append(f"{call.name}({call.arguments}) succeeded")
                        checkpoint.important_tool_results.append(observation[:500])
                else:
                    consecutive_failures += 1
                    if result.error:
                        run.errors.append(f"{call.name}:{result.error}")
                if task is not None:
                    task.observe(call.name, observation[:500], success=result.success)
                    task.current_step = len(run.steps)

            messages.append(Message.tool_results(outcomes))

            if consecutive_failures >= self.limits.max_consecutive_failures:
                return self._stop(
                    run,
                    started,
                    FAILURE_LIMIT,
                    self._partial_answer(run, "I couldn't get that working, sir."),
                )

        return self._stop(
            run,
            started,
            STEP_LIMIT,
            self._partial_answer(run, "I reached my step limit before finishing that, sir."),
        )

    # ------------------------------------------------------------------
    def _call_model(self, messages: list[Message], system: str, specs: list[ToolSpec], *, request_id: str = "", checkpoint: Any = None) -> ModelResponse:
        kwargs = {"system": system, "tools": specs, "effort": self.effort, "on_text": self.on_answer_text}
        if getattr(self.provider, "name", "") == "multi_model":
            kwargs.update(request_id=request_id, checkpoint=checkpoint)
        return self.provider.complete(messages, **kwargs)

    # ------------------------------------------------------------------
    def _parallel_safe(self, calls: list[ToolCall]) -> bool:
        """May these tool calls run at the same time?

        Only when every one of them is READ-ONLY and needs no exclusive
        resource. Read-only means it cannot change state another call in the
        batch might read; no exclusive resource means it cannot contend for
        the keyboard, the foreground window or a browser page. Together those
        are exactly the conditions under which running them concurrently
        cannot change any individual result -- which is the only kind of
        speed-up worth having.

        Anything else stays strictly sequential: a write, a desktop action, a
        browser step, an unknown tool, or the same call repeated (a repeat is
        a retry, not a batch, and the retry accounting depends on order).
        """
        if len(calls) < 2:
            return False
        signatures = set()
        for call in calls:
            definition = self.catalog.get(call.name)
            if definition is None or not definition.read_only or definition.exclusive_resource:
                return False
            # Read-only is not enough on its own for the desktop and browser
            # tools: their answers describe SHARED session state (which
            # window has focus, which page is loaded), so two of them at once
            # can observe a different desktop from the one the model is
            # reasoning about -- and the UIA/CDP layers underneath are not
            # designed to be driven from several threads at once.
            if definition.category in SESSION_AWARE_CATEGORIES or definition.name in SESSION_AWARE_TOOLS:
                return False
            signature = _signature(call)
            if signature in signatures:
                return False
            signatures.add(signature)
        return True

    def _execute_parallel(self, calls: list[ToolCall], cancellation_token: Any, run: AgentRun) -> dict[str, ToolResult]:
        """Run an eligible batch concurrently, keyed by call id.

        Results are returned as a mapping rather than a list so the caller's
        existing per-call bookkeeping (attempt counts, step records, ordering
        of `tool_result` blocks) is completely unchanged -- only WHEN each
        tool ran differs.
        """
        started = time.perf_counter()
        results: dict[str, ToolResult] = {}
        durations: list[float] = []
        workers = min(len(calls), self.max_parallel_tools)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jarvis-agent-tool") as pool:
            futures = {
                call.id: pool.submit(
                    self._execute_timed, call, cancellation_token, durations
                )
                for call in calls
            }
            for call_id, future in futures.items():
                results[call_id] = future.result()
        elapsed = (time.perf_counter() - started) * 1000
        run.parallel_batches += 1
        # Honest saving: what the same work would have cost sequentially,
        # minus what it actually cost. Never negative.
        run.parallel_saved_ms += max(0.0, sum(durations) - elapsed)
        log.info(
            "Ran %d independent read-only tools concurrently in %.0f ms (sequential would be ~%.0f ms): %s",
            len(calls), elapsed, sum(durations), [call.name for call in calls],
        )
        return results

    def _execute_timed(self, call: ToolCall, cancellation_token: Any, durations: list[float]) -> ToolResult:
        started = time.perf_counter()
        try:
            return self.catalog.execute(call.name, call.arguments, cancellation_token=cancellation_token)
        finally:
            durations.append((time.perf_counter() - started) * 1000)

    def _record_step(
        self,
        run: AgentRun,
        call: ToolCall,
        result: ToolResult | None,
        observation: str,
        attempt: int,
        thought: str,
    ) -> None:
        data = result.data if result is not None and isinstance(result.data, dict) else {}
        run.steps.append(
            StepRecord(
                index=len(run.steps),
                tool=call.name,
                arguments=_safe_arguments(call.arguments),
                success=bool(result.success) if result is not None else False,
                verified=bool(data.get("verified", False)),
                observation=observation[:2000],
                error=(result.error if result is not None else "refused_repeated_failure"),
                duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
                attempt=attempt,
                thought=(thought or "")[:500],
            )
        )

    def _conclude(self, run: AgentRun, started: float, response: ModelResponse) -> AgentRun:
        """Finish on a model turn that produced no tool call.

        Success is judged from the RUN, not from the model's tone: if the
        last thing any tool did was fail, this did not succeed, whatever
        the closing sentence says.
        """
        run.answer = response.text.strip() or "Done, sir."
        run.stop_reason = COMPLETED
        acting_steps = [step for step in run.steps if step.tool not in {"recall_memory"}]
        last_failed = bool(acting_steps) and not acting_steps[-1].success
        run.success = not last_failed
        # "Verified" is stricter: the FINAL observable state was
        # independently confirmed by the tool that produced it. Earlier
        # failures are normal in agentic work -- reproducing a bug is a
        # failing step on purpose -- so they do not disqualify a run whose
        # last step genuinely succeeded and was confirmed. A run with no
        # tool calls at all (a pure answer) is never marked verified.
        run.verified = bool(acting_steps) and acting_steps[-1].success and acting_steps[-1].verified
        run.duration_ms = (time.perf_counter() - started) * 1000
        self._emit("finished", run.describe())
        log.info("Agent run finished: %s", run.describe())
        return run

    def _stop(self, run: AgentRun, started: float, reason: str, answer: str) -> AgentRun:
        run.stop_reason = reason
        run.answer = answer
        run.success = False
        run.verified = False
        run.duration_ms = (time.perf_counter() - started) * 1000
        self._emit("finished", run.describe())
        log.info("Agent run stopped: %s", run.describe())
        return run

    def _partial_answer(self, run: AgentRun, prefix: str) -> str:
        """Report real partial progress instead of a bare failure."""
        done = [step for step in run.steps if step.success]
        if not done:
            return prefix
        last = done[-1]
        return f"{prefix} I got as far as {last.tool.replace('_', ' ')}."

    def _emit(self, stage: str, payload: dict[str, Any]) -> None:
        if self.progress is None:
            return
        try:
            self.progress(stage, payload)
        except Exception:
            log.exception("Agent progress callback failed")


def _is_cancelled(token: Any) -> bool:
    return token is not None and bool(getattr(token, "cancelled", False))


def _signature(call: ToolCall) -> str:
    try:
        arguments = json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        arguments = str(call.arguments)
    return f"{call.name}:{arguments}"


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Bound long argument values before they are stored in an episode."""
    safe: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and len(value) > 500:
            safe[key] = value[:500] + f"... [{len(value) - 500} more characters]"
        else:
            safe[key] = value
    return safe


#: How many entries of a list-shaped tool result the model actually reads.
#: A directory listing or a search result is evidence, but the 900th
#: filename is not: it costs input tokens on THIS step and on every step
#: after it, because the whole conversation is re-sent each turn. The count
#: and the continuation hint below preserve what the truncation removed --
#: the model can always ask for more with a narrower path or query.
MAX_LIST_ITEMS = 40


def _compact_list(key: str, value: Any) -> str:
    """Render a list-shaped result compactly, without hiding its size.

    Says how many there are, shows the first `MAX_LIST_ITEMS`, and tells the
    model exactly how to see the rest. Truncating silently would be the one
    unacceptable version of this: the model would reason as if it had seen
    everything.
    """
    if not isinstance(value, list) or len(value) <= MAX_LIST_ITEMS:
        return f"{key}: {json.dumps(value, default=str)[:2000]}"
    shown = json.dumps(value[:MAX_LIST_ITEMS], default=str)
    return (
        f"{key}: {len(value)} entries in total; first {MAX_LIST_ITEMS} shown: {shown}\n"
        f"({len(value) - MAX_LIST_ITEMS} more of the same kind are not listed. Repeating this same call "
        f"returns this same summary; use a more specific path or query if you actually need them.)"
    )


def _observation_text(result: ToolResult) -> str:
    """Turn a `ToolResult` into what the model should actually read.

    Structured fields the model can act on (stdout, contents, matches) are
    surfaced explicitly; a failure always leads with the error code so it
    cannot be mistaken for success.
    """
    data = result.data if isinstance(result.data, dict) else {}
    parts: list[str] = []
    if not result.success:
        parts.append(f"FAILED ({result.error or 'unknown_error'}): {result.message}")
    elif result.message:
        parts.append(result.message)

    for key in ("stdout", "stderr"):
        value = data.get(key)
        if value:
            parts.append(f"{key}:\n{value}")
    if data.get("exit_code") is not None:
        parts.append(f"exit_code: {data['exit_code']}")
    # Several tools already put their content in `message`. Send the file
    # body once: prefer the line-numbered form, and if EITHER form is
    # already present skip both, so the model never reads the same file
    # twice in one observation.
    contents = [str(data[key]) for key in ("numbered_contents", "contents") if data.get(key)]
    if contents and not any(item in parts for item in contents):
        parts.append(contents[0])
    for key in ("items", "matches", "files", "project_markers", "entry_points", "top_level_directories"):
        value = data.get(key)
        if value:
            parts.append(_compact_list(key, value))
    if not parts:
        parts.append("The tool completed but reported no detail.")
    return "\n".join(parts)


def _accumulate(run: AgentRun, response: ModelResponse) -> None:
    # `reported` means "every provider turn told us its token counts". The
    # first turn sets it; later turns can only take it away -- starting
    # from the AgentRun default (False, meaning "no calls yet, so unknown")
    # would otherwise make it permanently False.
    first_call = run.model_calls <= 1
    run.usage.reported = response.usage.reported if first_call else (run.usage.reported and response.usage.reported)
    run.usage.input_tokens += response.usage.input_tokens
    run.usage.output_tokens += response.usage.output_tokens
    run.usage.cache_creation_tokens += response.usage.cache_creation_tokens
    run.usage.cache_read_tokens += response.usage.cache_read_tokens
    if response.estimated_cost_usd is not None:
        run.estimated_cost_usd = round((run.estimated_cost_usd or 0.0) + response.estimated_cost_usd, 8)
    if response.model:
        run.model = response.model
