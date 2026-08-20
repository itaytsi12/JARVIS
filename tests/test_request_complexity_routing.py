"""Regression tests for the whole-request complexity guard.

Reproduces a real live-voice routing failure: the request

    "Inspect this JARVIS project and tell me how the main components are
     connected. Do not modify anything."

matched the router's `inspect (.+)` pattern, which captured the entire
sentence as an application name and produced
`inspect_window(app_name="this jarvis project and tell me how ...")`.
That single accessibility call could not satisfy the request and failed.

Two properties are locked down here:

- a keyword inside a complex sentence must NOT force a single local tool
  (`brain/request_complexity.py`), and
- the deterministic fast path for genuinely simple commands must be
  completely unaffected.
"""
from __future__ import annotations

import unittest

from brain.request_complexity import (
    assess_complexity,
    count_operations,
    looks_like_simple_target,
)
from brain.router import route_command
from voice.text_normalizer import normalize_transcript

# The exact sentence from the live failure.
LIVE_BUG_COMMAND = (
    "Inspect this JARVIS project and tell me how the main components are connected. "
    "Do not modify anything."
)

COMPLEX_REQUESTS = (
    LIVE_BUG_COMMAND,
    "inspect this project and explain how it works",
    "inspect my code and find the bug",
    "look through this repository and tell me how the components connect",
    "run this project and figure out why it fails",
    "open my project, inspect the error, fix it and test again",
)

# Simple commands whose existing deterministic route must not change.
SIMPLE_COMMANDS = {
    "open Spotify": "open_application",
    "volume down": "volume_down",
    "volume up": "volume_up",
    "calculate 527 * 93": "calculator",
    "inspect window": "inspect_window",
    "inspect the active window": "inspect_window",
    "describe this window": "inspect_window",
    "inspect notepad": "inspect_window",
    "list the controls in chrome": "inspect_window",
    "switch to chrome": "focus_application",
    "focus notepad": "focus_application",
    "press enter": "press_key",
    "close chrome": "close_application",
    "what time is it": "get_time",
}


class ComplexRequestsEscalateTests(unittest.TestCase):
    def test_the_live_bug_command_no_longer_routes_to_inspect_window(self):
        route = route_command(LIVE_BUG_COMMAND)
        self.assertNotEqual(route.get("tool"), "inspect_window")
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route.get("route_source"), "complexity_guard")

    def test_complex_requests_route_to_the_agent_runtime(self):
        for command in COMPLEX_REQUESTS:
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(
                    route["type"],
                    "agent_task",
                    f"expected agent escalation, got {route!r}",
                )

    def test_no_complex_request_is_reduced_to_a_single_local_tool(self):
        """The general property, independent of which tool keyword appears."""
        for command in COMPLEX_REQUESTS:
            with self.subTest(command=command):
                self.assertIsNone(route_command(command).get("tool"))

    def test_the_escalated_goal_keeps_the_whole_request(self):
        """The agent must receive the entire request, not a captured fragment."""
        route = route_command(LIVE_BUG_COMMAND)
        self.assertEqual(route["goal"], LIVE_BUG_COMMAND.strip())
        self.assertIn("Do not modify anything", route["goal"])

    def test_escalation_records_why_it_escalated(self):
        complexity = route_command(LIVE_BUG_COMMAND)["complexity"]
        self.assertTrue(complexity["is_complex"])
        self.assertTrue(complexity["requires_reasoning"])
        self.assertTrue(complexity["references_codebase"])
        self.assertTrue(complexity["has_constraint"])


class SimpleCommandsStayLocalTests(unittest.TestCase):
    def test_simple_commands_keep_their_deterministic_tool(self):
        for command, expected_tool in SIMPLE_COMMANDS.items():
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], "tool", f"{command!r} left the fast path: {route!r}")
                self.assertEqual(route["tool"], expected_tool)

    def test_no_simple_command_escalates_to_the_agent(self):
        for command in SIMPLE_COMMANDS:
            with self.subTest(command=command):
                self.assertNotEqual(route_command(command)["type"], "agent_task")

    def test_ordinary_multi_step_commands_still_use_the_local_paths(self):
        """Chained ACTIONS are not complexity: the planners still own these."""
        for command in ("open notepad and type hello", "open chrome and search for cats"):
            with self.subTest(command=command):
                self.assertIn(route_command(command)["type"], {"local_plan", "plan"})

    def test_a_short_target_that_mentions_code_stays_local(self):
        """A domain word alone must never be enough to escalate."""
        self.assertNotEqual(route_command("open vs code")["type"], "agent_task")
        self.assertFalse(assess_complexity("open vs code").is_complex)
        self.assertFalse(assess_complexity("save the file").is_complex)


class VoiceAndTypedPathsAgreeTests(unittest.TestCase):
    """The voice path normalizes the transcript first (wake word removal);
    both paths must reach the same routing decision."""

    def test_wake_word_prefixed_transcript_escalates_identically(self):
        spoken = f"Hey Jarvis, {LIVE_BUG_COMMAND[0].lower()}{LIVE_BUG_COMMAND[1:]}"
        normalized, _ = normalize_transcript(spoken)
        self.assertEqual(route_command(normalized)["type"], "agent_task")
        self.assertEqual(route_command(normalized)["type"], route_command(LIVE_BUG_COMMAND)["type"])

    def test_voice_normalized_simple_commands_still_route_locally(self):
        for command, expected_tool in SIMPLE_COMMANDS.items():
            with self.subTest(command=command):
                normalized, _ = normalize_transcript(f"Hey Jarvis, {command}")
                route = route_command(normalized)
                self.assertEqual(route["type"], "tool")
                self.assertEqual(route["tool"], expected_tool)


class CoverageGuardTests(unittest.TestCase):
    """`looks_like_simple_target` decides whether an open-ended capture
    actually captured a target rather than the rest of a sentence."""

    def test_real_targets_are_accepted(self):
        for target in ("notepad", "chrome", "the active window", "visual studio code", "this window"):
            with self.subTest(target=target):
                self.assertTrue(looks_like_simple_target(target))

    def test_sentence_fragments_are_rejected(self):
        for target in (
            "this jarvis project and tell me how the main components are connected",
            "my code and find the bug",
            "the project, then explain it",
            "this project and explain how it works",
            "",
        ):
            with self.subTest(target=target):
                self.assertFalse(looks_like_simple_target(target))


class ComplexityAssessmentTests(unittest.TestCase):
    def test_operations_are_counted_across_separators(self):
        self.assertEqual(count_operations("open Spotify"), 1)
        self.assertEqual(count_operations("open notepad and type hello"), 2)
        self.assertEqual(
            count_operations("open my project, inspect the error, fix it and test again"), 4
        )

    def test_reasoning_requests_about_software_are_complex(self):
        self.assertTrue(assess_complexity("explain how the router works in this repo").is_complex)

    def test_a_plain_factual_question_is_not_complex(self):
        """Ordinary questions must keep reaching the existing question route."""
        for question in (
            "who is the president of france",
            "what is the weather in san francisco tomorrow",
            "tell me what the capital of japan is",
        ):
            with self.subTest(question=question):
                self.assertFalse(assess_complexity(question).is_complex)

    def test_empty_input_is_handled(self):
        self.assertFalse(assess_complexity("").is_complex)
        self.assertFalse(assess_complexity("   ").is_complex)


if __name__ == "__main__":
    unittest.main()
