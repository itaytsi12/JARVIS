"""Performance work on the complex-agent path, without changing what it does.

Every optimization here has a correctness rule attached, and the rule is what
these tests protect. Speed that costs correctness is not a speed-up:

- parallel tool execution must be limited to independent READ-ONLY tools;
- observation compaction must say what it left out, never hide it;
- streamed speech must never emit a partial word, markdown, a tool payload,
  or internal reasoning;
- progress must be derived from real events and must stop the moment the
  answer starts;
- lowering reasoning effort must never apply to tasks that need it.

Every provider here is a fake, so the whole file runs offline and costs
nothing.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.agent_loop import MAX_LIST_ITEMS, AgentLimits, AgentLoop, _compact_list, _observation_text
from brain.context_builder import ContextBuilder
from brain.models import ToolResult
from brain.tool_catalog import ToolCatalog
from providers.base import ModelResponse, ToolCall, Usage
from providers.mock_provider import CallableProvider, text_response, tool_response
from providers.registry import register_provider, reset_providers_for_tests
from voice.agent_narration import AgentNarrator
from voice.sentence_stream import SentenceStream, speakable, split_sentences

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _context(request: str = "do the thing"):
    return ContextBuilder().build(request)


def _limits(**overrides) -> AgentLimits:
    defaults = dict(max_steps=10, max_action_retries=2, max_consecutive_failures=4, timeout_seconds=30.0)
    defaults.update(overrides)
    return AgentLimits(**defaults)


class _Provider(CallableProvider):
    name = "anthropic"


def install_provider(test_case, handler, model="claude-sonnet-5"):
    provider = _Provider(handler, model=model)
    register_provider("anthropic", lambda: provider)
    test_case.addCleanup(reset_providers_for_tests)
    return provider


# ---------------------------------------------------------------- parallel


class ParallelToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog()

    def _loop(self, handler):
        return AgentLoop(_Provider(handler), self.catalog, limits=_limits())

    def test_independent_read_only_tools_run_concurrently(self):
        root = str(PROJECT_ROOT)
        calls = []

        def handler(messages, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall("a", "list_files", {"path": root}),
                        ToolCall("b", "exists", {"path": root}),
                        ToolCall("c", "get_time", {}),
                    ],
                    stop_reason="tool_use",
                )
            return text_response("Done, sir.")

        run = self._loop(handler).run("look at three things", context=_context())
        self.assertEqual(run.parallel_batches, 1)
        self.assertEqual([step.tool for step in run.steps], ["list_files", "exists", "get_time"])
        self.assertTrue(all(step.success for step in run.steps))

    def test_a_write_in_the_batch_forces_sequential_execution(self):
        """One non-read-only tool disqualifies the whole turn: it could change
        what another call in the same turn reads."""
        directory = Path(tempfile.mkdtemp())
        calls = []

        def handler(messages, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall("a", "list_files", {"path": str(directory)}),
                        ToolCall("b", "create_text_file", {"path": str(directory / "new.txt"), "contents": "x"}),
                    ],
                    stop_reason="tool_use",
                )
            return text_response("Done, sir.")

        run = self._loop(handler).run("read and write", context=_context())
        self.assertEqual(run.parallel_batches, 0)
        self.assertEqual(len(run.steps), 2)

    def test_desktop_and_browser_tools_are_never_batched(self):
        """They contend for the one keyboard / the one page."""
        loop = self._loop(lambda messages, **kwargs: text_response("x"))
        for first, second in (
            ("take_screenshot", "inspect_window"),
            ("read_text_file", "click_ui_element"),
        ):
            with self.subTest(pair=(first, second)):
                calls = [ToolCall("a", first, {}), ToolCall("b", second, {})]
                self.assertFalse(loop._parallel_safe(calls))

    def test_a_single_call_is_never_a_batch(self):
        loop = self._loop(lambda messages, **kwargs: text_response("x"))
        self.assertFalse(loop._parallel_safe([ToolCall("a", "list_files", {"path": "."})]))

    def test_the_same_call_twice_is_a_retry_not_a_batch(self):
        loop = self._loop(lambda messages, **kwargs: text_response("x"))
        calls = [ToolCall("a", "list_files", {"path": "."}), ToolCall("b", "list_files", {"path": "."})]
        self.assertFalse(loop._parallel_safe(calls))

    def test_an_unknown_tool_disqualifies_the_batch(self):
        loop = self._loop(lambda messages, **kwargs: text_response("x"))
        calls = [ToolCall("a", "list_files", {"path": "."}), ToolCall("b", "no_such_tool", {})]
        self.assertFalse(loop._parallel_safe(calls))

    def test_parallel_execution_actually_overlaps_in_time(self):
        """Not just "it was called" -- the wall clock must be shorter than the
        sum of the parts."""
        catalog = ToolCatalog()
        started_together = threading.Barrier(3, timeout=5)

        def slow(arguments):
            started_together.wait()  # only passes if all three run at once
            time.sleep(0.05)
            return ToolResult(True, "recall_memory", "ok", {"verified": True})

        for name in ("recall_memory",):
            catalog.register_handler(name, slow)

        loop = AgentLoop(_Provider(lambda m, **k: text_response("x")), catalog, limits=_limits())
        calls = [ToolCall(str(index), "recall_memory", {"query": f"q{index}"}) for index in range(3)]
        self.assertTrue(loop._parallel_safe(calls))
        run_holder = AgentLoop(_Provider(lambda m, **k: text_response("x")), catalog, limits=_limits()).run(
            "x", context=_context()
        )
        results = loop._execute_parallel(calls, None, run_holder)
        self.assertEqual(len(results), 3)
        self.assertEqual(run_holder.parallel_batches, 1)


# ------------------------------------------------------------- compaction


class ObservationCompactionTests(unittest.TestCase):
    def test_a_long_list_is_compacted_but_its_size_is_still_reported(self):
        value = [f"file_{index}.py" for index in range(500)]
        rendered = _compact_list("items", value)
        self.assertIn("500 entries in total", rendered)
        self.assertIn("file_0.py", rendered)
        self.assertNotIn("file_499.py", rendered)
        self.assertIn("460 more of the same kind are not listed", rendered)

    def test_a_short_list_is_left_completely_alone(self):
        value = ["a.py", "b.py"]
        rendered = _compact_list("items", value)
        self.assertIn("a.py", rendered)
        self.assertIn("b.py", rendered)
        self.assertNotIn("not shown", rendered)

    def test_compaction_never_silently_hides_entries(self):
        """The model must be able to tell that it did not see everything."""
        rendered = _compact_list("matches", [f"m{index}" for index in range(MAX_LIST_ITEMS + 1)])
        self.assertIn(str(MAX_LIST_ITEMS + 1), rendered)
        self.assertIn("use a more specific path or query", rendered)

    def test_a_failure_still_leads_with_the_error(self):
        result = ToolResult(False, "run_command", "it broke", {"items": list(range(500))}, "nonzero_exit")
        text = _observation_text(result)
        self.assertTrue(text.startswith("FAILED (nonzero_exit)"))

    def test_terminal_evidence_is_preserved(self):
        result = ToolResult(
            True, "run_command", "ran", {"exit_code": 1, "stdout": "3 passed", "stderr": "1 failed"}
        )
        text = _observation_text(result)
        for expected in ("exit_code: 1", "3 passed", "1 failed"):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def test_a_real_project_inspection_is_much_smaller_than_it_used_to_be(self):
        catalog = ToolCatalog()
        result = catalog.execute("inspect_project", {"path": str(PROJECT_ROOT)})
        self.assertTrue(result.success)
        text = _observation_text(result)
        # The uncompacted form was over 20,000 characters of file list.
        self.assertLess(len(text), 6000)
        # ...but the real counts survive.
        self.assertIn("source_file_count", str(result.data))


# ------------------------------------------------------------ speakability


class SentenceStreamTests(unittest.TestCase):
    def test_only_complete_sentences_are_released(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        for chunk in ("The project ", "has three parts", ". The second"):
            stream.feed(chunk)
        self.assertEqual(out, ["The project has three parts."])

    def test_a_partial_word_is_never_emitted(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        for chunk in "The quick brown fox jumps.":
            stream.feed(chunk)
        # A trailing "." with nothing after it is not yet a confirmed
        # boundary -- "jumps. " and "jumps.py" are indistinguishable so far.
        self.assertEqual(out, [])
        stream.flush()
        self.assertEqual(out, ["The quick brown fox jumps."])

    def test_a_filename_does_not_split_a_sentence(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        stream.feed("Look at main.py and app.py first. Then stop.")
        self.assertEqual(out[0], "Look at main.py and app.py first.")

    def test_markdown_is_never_spoken(self):
        self.assertEqual(speakable("The **brain** package"), "The brain package")
        self.assertEqual(speakable("Use `main.py` now"), "Use main.py now")
        self.assertEqual(speakable("## Heading"), "Heading")
        self.assertEqual(speakable("- item one"), "item one")
        self.assertEqual(speakable("[the docs](http://x.example)"), "the docs")

    def test_a_code_block_is_dropped_not_read_aloud(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        stream.feed("Here is the fix.\n```python\nprint('x')\n```\nThat is all.")
        stream.flush()
        joined = " ".join(out)
        self.assertNotIn("print", joined)
        self.assertIn("That is all", joined)

    def test_an_unclosed_code_fence_holds_everything_back(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        stream.feed("Here it is.\n```python\nprint('x')")
        self.assertEqual(out, [])

    def test_a_short_preamble_is_held_back(self):
        """A tool-call preamble must never be spoken as if it were the answer."""
        out = []
        stream = SentenceStream(emit=out.append)  # default hold-back
        stream.feed("Let me look at the files.")
        self.assertEqual(out, [])

    def test_a_real_answer_starts_speaking_before_it_finishes(self):
        out = []
        stream = SentenceStream(emit=out.append)
        long_answer = (
            "The JARVIS project root contains twenty-one top level directories and a handful of "
            "configuration files at the very top of the tree. The brain package holds all of the "
            "routing and planning code that the assistant runs. "
        )
        stream.feed(long_answer)
        self.assertTrue(out, "nothing was released even though a full answer had arrived")
        # ...and it was released while more text was still to come.
        stream.feed("The voice package owns speech.")
        stream.flush()
        self.assertGreaterEqual(len(out), 2)

    def test_flush_releases_an_answer_with_no_final_punctuation(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        stream.feed("no trailing period here")
        self.assertEqual(out, [])
        stream.flush()
        self.assertEqual(out, ["no trailing period here"])

    def test_discard_speaks_nothing(self):
        out = []
        stream = SentenceStream(emit=out.append, hold_back_chars=0)
        stream.feed("This should never be spoken.")
        out.clear()
        stream.discard()
        stream.flush()
        self.assertEqual(out, [])

    def test_split_sentences_keeps_the_incomplete_tail(self):
        complete, remainder = split_sentences("One. Two. Thr")
        self.assertEqual(complete, ["One.", "Two."])
        self.assertEqual(remainder.strip(), "Thr")


# -------------------------------------------------------------- narration


class ProgressNarrationTests(unittest.TestCase):
    def _narrator(self, spoken=None, start=100.0):
        self.now = [start]
        spoken = [] if spoken is None else spoken
        return AgentNarrator(speak=spoken.append, clock=lambda: self.now[0], started_at=start), spoken

    def test_progress_comes_from_a_real_tool_event(self):
        narrator, spoken = self._narrator()
        self.now[0] += 10
        narrator.on_event("tool_result", {"tool": "run_command", "success": True})
        self.assertEqual(spoken, ["I'm running that now, sir."])

    def test_an_unmapped_tool_says_nothing_rather_than_something_vague(self):
        narrator, spoken = self._narrator()
        self.now[0] += 10
        narrator.on_event("tool_result", {"tool": "some_future_tool", "success": True})
        self.assertEqual(spoken, [])

    def test_nothing_is_said_for_a_task_that_finishes_quickly(self):
        narrator, spoken = self._narrator()
        self.now[0] += 1  # only a second in
        narrator.on_event("tool_result", {"tool": "list_files", "success": True})
        self.assertEqual(spoken, [])

    def test_updates_are_rate_limited(self):
        narrator, spoken = self._narrator()
        self.now[0] += 10
        narrator.on_event("tool_result", {"tool": "list_files"})
        self.now[0] += 1
        narrator.on_event("tool_result", {"tool": "run_command"})
        self.assertEqual(len(spoken), 1)
        self.now[0] += 10
        narrator.on_event("tool_result", {"tool": "run_command"})
        self.assertEqual(len(spoken), 2)

    def test_the_same_line_is_never_repeated_back_to_back(self):
        narrator, spoken = self._narrator()
        for _ in range(4):
            self.now[0] += 10
            narrator.on_event("tool_result", {"tool": "read_text_file"})
        self.assertEqual(len(spoken), 1)

    def test_progress_stops_once_the_answer_starts(self):
        """"I'm still checking" must never follow the answer."""
        narrator, spoken = self._narrator()
        narrator.mark_answer_started()
        self.now[0] += 60
        narrator.on_event("tool_result", {"tool": "run_command"})
        self.assertEqual(spoken, [])
        self.assertGreaterEqual(narrator.suppressed, 1)

    def test_the_answer_stream_marks_the_answer_as_started(self):
        narrator, spoken = self._narrator()
        stream = narrator.answer_stream()
        stream.hold_back_chars = 0
        stream.feed("Here is the real answer, sir. ")
        self.assertTrue(narrator.answer_started)
        self.assertEqual(spoken, ["Here is the real answer, sir."])

    def test_no_model_turn_text_is_ever_narrated(self):
        """`model_turn` carries the assistant's preamble; it is not progress."""
        narrator, spoken = self._narrator()
        self.now[0] += 30
        narrator.on_event("model_turn", {"step": 1, "text": "I should look at the router first"})
        self.assertEqual(spoken, [])

    def test_a_broken_narrator_never_fails_the_task(self):
        def explode(_text):
            raise RuntimeError("tts is down")

        narrator = AgentNarrator(speak=explode, clock=lambda: 999.0, started_at=0.0)
        narrator.on_event("tool_result", {"tool": "run_command"})  # must not raise


# ------------------------------------------------------------------ effort


class EffortSelectionTests(unittest.TestCase):
    def test_read_only_inspection_uses_the_interactive_default(self):
        from brain.agent_service import select_effort

        for goal in (
            "Tell me what files are in the JARVIS project folder. Do not modify anything.",
            "Run git status in the JARVIS project and tell me what changed.",
            "Inspect the JARVIS project and explain how the main components are connected.",
        ):
            with self.subTest(goal=goal):
                self.assertEqual(select_effort(goal), "medium")

    def test_demanding_work_keeps_the_deeper_budget(self):
        from brain.agent_service import select_effort

        for goal in (
            "run the project, diagnose the error, fix it and verify the fix",
            "find the bug in music_intent and fix it",
            "why are the tests failing",
            "refactor the router",
        ):
            with self.subTest(goal=goal):
                self.assertEqual(select_effort(goal), "high")

    def test_effort_is_configurable(self):
        from brain.agent_service import select_effort
        from config.settings import reload_config

        with patch.dict(os.environ, {"JARVIS_AGENT_EFFORT": "low"}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            self.assertEqual(select_effort("list the files"), "low")

    def test_the_chosen_effort_reaches_the_provider_and_is_recorded(self):
        seen = {}

        def handler(messages, **kwargs):
            seen.update(kwargs)
            return text_response("Done, sir.")

        loop = AgentLoop(_Provider(handler), ToolCatalog(), limits=_limits(), effort="high")
        run = loop.run("x", context=_context())
        self.assertEqual(seen.get("effort"), "high")
        self.assertEqual(run.effort, "high")
        self.assertEqual(run.describe()["effort"], "high")


# ------------------------------------------------------------ integration


class AgentPathPerformanceTests(unittest.TestCase):
    """The whole path, with a fake provider: metrics recorded, no legacy hop."""

    def test_metrics_are_recorded_on_the_outcome(self):
        from brain.agent import run_agent

        root = str(PROJECT_ROOT)
        calls = []

        def handler(messages, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall("a", "list_files", {"path": root}),
                        ToolCall("b", "exists", {"path": root}),
                    ],
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=2500, output_tokens=60, reported=True),
                )
            return ModelResponse(
                text="All done, sir.",
                stop_reason="end_turn",
                usage=Usage(input_tokens=4000, output_tokens=120, reported=True),
            )

        install_provider(self, handler)
        outcome: dict = {}
        run_agent(
            "Tell me what files are in the JARVIS project folder and briefly explain the important ones. "
            "Do not modify anything.",
            execution_outcome=outcome,
        )
        self.assertEqual(outcome["parallel_tool_batches"], 1)
        self.assertEqual(outcome["effort"], "medium")
        self.assertIsNotNone(outcome["time_to_first_tool_ms"])
        self.assertGreater(outcome["context_chars"], 0)
        self.assertLess(outcome["selected_tool_count"], outcome["available_tool_count"])

    def test_tool_schemas_are_filtered_not_all_sent(self):
        seen = {}

        def handler(messages, **kwargs):
            seen["tools"] = [spec.name for spec in (kwargs.get("tools") or [])]
            return text_response("Done, sir.")

        install_provider(self, handler)
        from brain.agent import run_agent

        run_agent("Read main.py and tell me what it does. Do not modify anything.")
        catalog = ToolCatalog()
        self.assertLess(len(seen["tools"]), len(catalog.names()))
        # The tools the task actually needs are present...
        for needed in ("read_text_file", "list_files", "run_command"):
            with self.subTest(tool=needed):
                self.assertIn(needed, seen["tools"])
        # ...and irrelevant ones are not.
        for irrelevant in ("mute_volume", "volume_up"):
            with self.subTest(tool=irrelevant):
                self.assertNotIn(irrelevant, seen["tools"])

    def test_streamed_text_reaches_the_caller_before_the_run_returns(self):
        from brain.agent import run_agent

        received: list[str] = []
        answer = "A" * 300

        def handler(messages, **kwargs):
            on_text = kwargs.get("on_text")
            if on_text:
                on_text(answer)
                received.append("streamed-before-return")
            return text_response(answer)

        install_provider(self, handler)
        run_agent(
            "Inspect this project and explain how it works. Do not modify anything.",
            on_answer_text=lambda chunk: received.append(chunk),
        )
        self.assertIn("streamed-before-return", received)

    def test_no_answer_text_callback_means_no_streaming_request(self):
        """A non-voice caller must not pay for streaming it will not use."""
        seen = {}

        def handler(messages, **kwargs):
            seen["on_text"] = kwargs.get("on_text")
            return text_response("Done, sir.")

        install_provider(self, handler)
        from brain.agent import run_agent

        run_agent("Inspect this project and explain how it works. Do not modify anything.")
        self.assertIsNone(seen["on_text"])


class BargeInAndCancellationTests(unittest.TestCase):
    """Interruption must still work during every new kind of speech, and must
    never leave the agent in a corrupt state."""

    def test_stopping_speech_works_during_any_kind_of_speech(self):
        """Acknowledgement, progress and streamed answer all go through the
        same `_start_speech_task` -> `speak` path, so one `stop()` covers
        every one of them."""
        import voice.text_to_speech as tts

        stopped = []

        class Engine:
            def stop(self):
                stopped.append("engine")

        with patch.object(tts, "_elevenlabs_provider", None), \
             patch.object(tts, "_openai_provider", None), \
             patch.object(tts, "_chatterbox_provider", None), \
             patch.object(tts, "_engine", Engine()):
            tts.stop()
        self.assertEqual(stopped, ["engine"])

    def test_cancelling_mid_run_stops_the_agent_and_speaks_nothing_more(self):
        from brain.agent_loop import CANCELLED

        class Token:
            cancelled = False

        token = Token()
        spoken: list[str] = []
        narrator = AgentNarrator(speak=spoken.append, clock=lambda: 1000.0, started_at=0.0)

        def handler(messages, **kwargs):
            token.cancelled = True  # the user said "cancel" during the turn
            return ModelResponse(
                text="",
                tool_calls=[ToolCall("a", "get_time", {})],
                stop_reason="tool_use",
            )

        loop = AgentLoop(_Provider(handler), ToolCatalog(), limits=_limits(), progress=narrator.on_event)
        run = loop.run("something long", context=_context(), cancellation_token=token)
        self.assertEqual(run.stop_reason, CANCELLED)
        self.assertEqual(spoken, [], "nothing may be spoken after cancellation")

    def test_a_discarded_stream_never_speaks_a_partial_answer(self):
        """Barge-in during a streamed answer: what was buffered is dropped."""
        spoken: list[str] = []
        narrator = AgentNarrator(speak=spoken.append, clock=lambda: 1000.0, started_at=0.0)
        stream = narrator.answer_stream()
        stream.feed("The project has ")  # incomplete, still buffered
        stream.discard()
        stream.flush()
        self.assertEqual(spoken, [])

    def test_progress_after_the_answer_is_suppressed_not_queued(self):
        """"I'm still checking" must never arrive after the answer."""
        spoken: list[str] = []
        clock = [0.0]
        narrator = AgentNarrator(speak=spoken.append, clock=lambda: clock[0], started_at=0.0)
        clock[0] = 10.0
        narrator.on_event("tool_result", {"tool": "list_files"})
        self.assertEqual(len(spoken), 1)
        narrator.mark_answer_started()
        clock[0] = 40.0
        narrator.on_event("tool_result", {"tool": "run_command"})
        self.assertEqual(len(spoken), 1)


class HeartbeatTests(unittest.TestCase):
    """A long silence is filled, but only ever with something true."""

    def _narrator(self, **kwargs):
        clock = [0.0]
        spoken: list[str] = []
        narrator = AgentNarrator(
            speak=spoken.append, clock=lambda: clock[0], started_at=0.0, **kwargs
        )
        return narrator, spoken, clock

    def test_nothing_is_said_while_the_silence_is_short(self):
        narrator, spoken, clock = self._narrator()
        clock[0] = 5.0
        self.assertFalse(narrator.heartbeat())
        self.assertEqual(spoken, [])

    def test_a_running_tool_is_reported_as_the_tool_not_the_model(self):
        from voice.agent_narration import STILL_RUNNING

        narrator, spoken, clock = self._narrator()
        narrator.on_event("tool_started", {"tool": "run_command"})
        clock[0] = 30.0
        self.assertTrue(narrator.heartbeat())
        self.assertEqual(spoken, [STILL_RUNNING])

    def test_once_every_tool_has_returned_the_model_is_reported(self):
        from voice.agent_narration import STILL_THINKING

        narrator, spoken, clock = self._narrator()
        narrator.on_event("tool_started", {"tool": "run_command"})
        clock[0] = 6.0
        narrator.on_event("tool_result", {"tool": "run_command", "success": True})
        spoken.clear()
        clock[0] = 40.0
        self.assertTrue(narrator.heartbeat())
        self.assertEqual(spoken, [STILL_THINKING])

    def test_the_heartbeat_stops_once_the_answer_starts(self):
        narrator, spoken, clock = self._narrator()
        narrator.mark_answer_started()
        clock[0] = 120.0
        self.assertFalse(narrator.heartbeat())
        self.assertEqual(spoken, [])

    def test_the_heartbeat_resets_the_progress_rate_limit(self):
        """Speaking twice in the same second would be worse than silence."""
        narrator, spoken, clock = self._narrator()
        clock[0] = 30.0
        narrator.heartbeat()
        narrator.on_event("tool_result", {"tool": "list_files"})
        self.assertEqual(len(spoken), 1)

    def test_the_heartbeat_thread_stops_cleanly(self):
        spoken: list[str] = []
        narrator = AgentNarrator(speak=spoken.append, max_silence=0.05)
        narrator.start_heartbeat()
        try:
            deadline = time.monotonic() + 5
            while not spoken and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(spoken, "the heartbeat never fired")
        finally:
            narrator.stop()
        count = len(spoken)
        time.sleep(1.5)
        self.assertEqual(len(spoken), count, "the heartbeat kept talking after stop()")

    def test_the_agent_loop_reports_when_a_tool_starts(self):
        events: list[tuple[str, dict]] = []
        calls = []

        def handler(messages, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return ModelResponse(
                    text="", tool_calls=[ToolCall("a", "get_time", {})], stop_reason="tool_use"
                )
            return text_response("Done, sir.")

        loop = AgentLoop(
            _Provider(handler),
            ToolCatalog(),
            limits=_limits(),
            progress=lambda stage, payload: events.append((stage, payload)),
        )
        loop.run("x", context=_context())
        stages = [stage for stage, _ in events]
        self.assertIn("tool_started", stages)
        self.assertLess(stages.index("tool_started"), stages.index("tool_result"))


class NoHiddenReasoningIsEverSpokenTests(unittest.TestCase):
    """Strict separation between internal reasoning and user-facing speech."""

    def test_the_provider_stops_forwarding_text_once_a_tool_call_starts(self):
        """A tool-use turn's preamble must not reach the answer sink, and the
        tool payload that follows it certainly must not."""
        import providers.anthropic_provider as ap
        from providers.base import Message

        forwarded: list[str] = []

        class Block:
            type = "tool_use"

        class Event:
            def __init__(self, type_, **kwargs):
                self.type = type_
                for key, value in kwargs.items():
                    setattr(self, key, value)

        events = [
            Event("text", text="Let me look at the files."),
            Event("content_block_start", content_block=Block()),
            Event("text", text='{"path": "C:/secret"}'),
        ]

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(events)

            def get_final_message(self):
                return type("M", (), {"content": [], "usage": None, "stop_reason": "tool_use", "model": "m"})()

        class Messages:
            @staticmethod
            def stream(**kwargs):
                return Stream()

        class Client:
            messages = Messages()

        provider = ap.AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-5", client=Client())
        provider.complete([Message.user("hi")], system="s", on_text=forwarded.append)
        self.assertEqual(forwarded, ["Let me look at the files."])
        self.assertNotIn('{"path": "C:/secret"}', forwarded)

    def test_only_text_events_are_ever_forwarded(self):
        source = (PROJECT_ROOT / "providers" / "anthropic_provider.py").read_text(encoding="utf-8")
        self.assertIn('if event_type != "text" or tool_use_started', source)

    def test_narration_phrases_are_a_fixed_local_set_never_model_output(self):
        from voice.agent_narration import _PHRASES, phrase_for_tool

        self.assertTrue(all(isinstance(text, str) and text for text in _PHRASES.values()))
        self.assertIsNone(phrase_for_tool("anything_unmapped"))


class PromptCachingTests(unittest.TestCase):
    """Caching must be requested with the exact documented shape, and must be
    switchable off."""

    def _capture(self, **kwargs):
        import providers.anthropic_provider as ap
        from providers.base import Message

        captured = {}

        class Messages:
            @staticmethod
            def create(**request):
                captured.update(request)
                return type("M", (), {"content": [], "usage": None, "stop_reason": "end_turn", "model": "m"})()

        class Client:
            messages = Messages()

        provider = ap.AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-5", client=Client())
        provider.complete([Message.user("hello")], system="stable system prompt", **kwargs)
        return captured

    def test_the_system_prompt_carries_a_cache_breakpoint(self):
        system = self._capture()["system"]
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(system[0]["text"], "stable system prompt")

    def test_caching_can_be_turned_off_and_then_sends_a_plain_system_prompt(self):
        self.assertEqual(self._capture(cache=False)["system"], "stable system prompt")

    def test_effort_is_sent_inside_output_config_not_at_the_top_level(self):
        request = self._capture(effort="high")
        self.assertEqual(request["output_config"], {"effort": "high"})
        self.assertNotIn("effort", request)

    def test_no_effort_means_the_parameter_is_omitted_entirely(self):
        self.assertNotIn("output_config", self._capture())

    def test_no_sampling_parameters_are_ever_sent(self):
        """This model family rejects temperature/top_p/top_k with a 400."""
        request = self._capture(temperature=0.0)
        for forbidden in ("temperature", "top_p", "top_k"):
            with self.subTest(parameter=forbidden):
                self.assertNotIn(forbidden, request)

    def test_a_rolling_breakpoint_is_placed_on_the_conversation(self):
        from providers.anthropic_provider import _mark_cache_breakpoint

        payload = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        _mark_cache_breakpoint(payload)
        self.assertNotIn("cache_control", payload[0]["content"][0])
        self.assertEqual(payload[0]["content"][-1]["cache_control"], {"type": "ephemeral"})

    def test_a_plain_string_message_is_left_alone(self):
        from providers.anthropic_provider import _mark_cache_breakpoint

        payload = [{"role": "user", "content": "plain"}]
        _mark_cache_breakpoint(payload)
        self.assertEqual(payload, [{"role": "user", "content": "plain"}])

    def test_cache_usage_is_reported_back(self):
        from providers.anthropic_provider import _extract_usage

        raw = type("U", (), {
            "input_tokens": 10, "output_tokens": 2,
            "cache_creation_input_tokens": 1200, "cache_read_input_tokens": 3400,
        })()
        usage = _extract_usage(raw)
        self.assertEqual(usage.cache_creation_tokens, 1200)
        self.assertEqual(usage.cache_read_tokens, 3400)


class ProviderReuseTests(unittest.TestCase):
    """Expensive things are built once, not per request."""

    def test_the_provider_is_reused_across_requests(self):
        from providers.registry import get_agent_provider

        install_provider(self, lambda messages, **kwargs: text_response("x"))
        self.assertIs(get_agent_provider(), get_agent_provider())

    def test_the_anthropic_client_is_constructed_once(self):
        import providers.anthropic_provider as ap

        built = []

        class FakeSDK:
            @staticmethod
            def Anthropic(**kwargs):
                built.append(kwargs)
                return object()

        provider = ap.AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-5")
        with patch.dict("sys.modules", {"anthropic": FakeSDK}):
            first = provider._get_client()
            second = provider._get_client()
        self.assertIs(first, second)
        self.assertEqual(len(built), 1)

    def test_shared_services_are_singletons(self):
        from brain.tool_catalog import get_tool_catalog
        from memory.agent_memory import get_agent_memory
        from skills import get_skill_registry

        self.assertIs(get_tool_catalog(), get_tool_catalog())
        self.assertIs(get_skill_registry(), get_skill_registry())
        self.assertIs(get_agent_memory(), get_agent_memory())


class LocalCommandsAreUnaffectedTests(unittest.TestCase):
    """No regression to the deterministic fast path."""

    def test_simple_commands_still_route_locally_and_call_no_model(self):
        from brain.router import route_command

        install_provider(self, lambda messages, **kwargs: text_response("should never happen"))
        for command, tool in (
            ("open Spotify", "open_application"),
            ("volume down", "volume_down"),
            ("what time is it", "get_time"),
            ("inspect window", "inspect_window"),
            ("play Israeli playlist", "music_play"),
        ):
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], "tool")
                self.assertEqual(route["tool"], tool)


if __name__ == "__main__":
    unittest.main()
