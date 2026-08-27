"""Regression tests for two conversational-reference gaps found by exercising
the merged context resolver against the commands users actually say.

Both were general resolver gaps rather than missing special cases:

- "run it again" was not recognised as a replay at all. Only the bare forms
  ("do it again", "try again") matched, so a verb in front defeated it and
  the request fell through to a full agent escalation.
- "open that project" named no project, so it missed the project route and
  reached the generic application fallback, which tried to launch an
  application literally called "that project".
"""
from __future__ import annotations

import time
import unittest

from brain.context_resolver import classify_reference_shape
from brain.router import route_command
from brain.session_context import SessionContext


def _with_project(name="jarvis", path=r"C:\Users\Ori\Desktop\jarvis") -> SessionContext:
    context = SessionContext()
    context.bump_turn()
    context.last_project_name = name
    context.last_project_path = path
    context.project_updated_at = time.monotonic()
    return context


def _with_last_command() -> SessionContext:
    context = SessionContext()
    context.bump_turn()
    context.record_app_event("spotify", "opened")
    context.record_command("open_application", {"app_name": "spotify"}, "open spotify")
    return context


class ReplayShapeTests(unittest.TestCase):
    def test_a_verb_before_it_again_is_still_a_replay(self):
        for phrase in ("run it again", "play it again", "open it again", "try it again", "execute it again"):
            self.assertEqual(classify_reference_shape(phrase), "again", phrase)

    def test_the_original_bare_forms_still_work(self):
        for phrase in ("do it again", "try again", "again", "repeat that", "once more"):
            self.assertEqual(classify_reference_shape(phrase), "again", phrase)

    def test_a_command_that_merely_ends_in_again_is_not_a_replay(self):
        # "run the tests again" names its own subject and is a real command,
        # not a reference to the previous one.
        self.assertIsNone(classify_reference_shape("run the tests again"))

    def test_run_it_again_replays_the_remembered_command(self):
        route = route_command("run it again", context=_with_last_command())
        self.assertEqual(route["route_source"], "context_replay")
        self.assertEqual(route["tool"], "open_application")
        self.assertEqual(route["arguments"], {"app_name": "spotify"})


class DeicticProjectTests(unittest.TestCase):
    def test_open_that_project_resolves_from_session_state(self):
        route = route_command("open that project", context=_with_project())
        self.assertEqual(route["tool"], "open_path")
        self.assertEqual(route["route_source"], "context_project_open")
        self.assertEqual(route["arguments"]["path"], r"C:\Users\Ori\Desktop\jarvis")

    def test_every_deictic_form_resolves_the_same_way(self):
        for phrase in ("open that project", "open the project", "open this project"):
            route = route_command(phrase, context=_with_project())
            self.assertEqual(route.get("route_source"), "context_project_open", phrase)

    def test_a_named_project_is_unaffected(self):
        route = route_command("open my jarvis project", context=_with_project())
        self.assertEqual(route["route_source"], "context_project_open")
        self.assertEqual(route["arguments"]["path"], r"C:\Users\Ori\Desktop\jarvis")

    def test_a_deictic_project_with_no_project_in_context_is_not_invented(self):
        # It must fall through rather than resolve to some arbitrary path.
        route = route_command("open that project", context=SessionContext())
        self.assertNotEqual(route.get("route_source"), "context_project_open")
        self.assertNotEqual(route.get("tool"), "open_path")


if __name__ == "__main__":
    unittest.main()
