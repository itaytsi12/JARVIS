"""Routing: the fast local path stays fast, and escalation only happens
when the local layer genuinely cannot resolve a request.
"""
import os
import time
import unittest
from unittest.mock import patch

import brain.agent as agent_module
from brain.router import route_command
from config.settings import reload_config


FAST_COMMANDS = {
    "open spotify": ("tool", "open_application"),
    "open youtube": ("tool", "open_website"),
    "volume down": ("tool", "volume_down"),
    "mute": ("tool", "mute_volume"),
    "take a screenshot": ("tool", "take_screenshot"),
    "what time is it": ("tool", "get_time"),
    "calculate 527 * 93": ("tool", "calculator"),
    "type hello": ("tool", "type_text"),
    "close notepad": ("tool", "close_application"),
    "show desktop": ("tool", "show_desktop"),
}


class FastPathTests(unittest.TestCase):
    """Backwards compatibility: the commands that worked before still work,
    still route locally, and still never involve a model."""

    def test_simple_commands_route_to_a_single_local_tool(self):
        for command, (route_type, tool) in FAST_COMMANDS.items():
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], route_type)
                self.assertEqual(route["tool"], tool)

    def test_routing_a_simple_command_is_fast(self):
        started = time.perf_counter()
        for command in FAST_COMMANDS:
            route_command(command)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertLess(elapsed_ms, 500, "local routing should not take hundreds of milliseconds")

    def test_a_deterministic_route_never_reaches_the_agent(self):
        with patch.object(agent_module, "_run_agent_with_loop") as escalate:
            with patch.object(agent_module, "_agent_escalation_available", return_value=True):
                agent_module.run_agent("what time is it")
        escalate.assert_not_called()

    def test_control_commands_stay_deterministic(self):
        self.assertEqual(route_command("cancel")["type"], "cancel_read_only_task")
        self.assertEqual(route_command("what are you doing")["type"], "task_status")


class MemoryRouteTests(unittest.TestCase):
    def test_an_explicit_remember_instruction_routes_locally(self):
        route = route_command("remember that my main project is at C:/dev/jarvis")
        self.assertEqual(route["type"], "remember")
        self.assertEqual(route["text"], "my main project is at C:/dev/jarvis")

    def test_remember_is_handled_without_a_model_and_stores_the_fact(self):
        from memory.agent_memory import get_agent_memory

        before = get_agent_memory().long_term.count()
        response = agent_module.run_agent("remember that I use vscode as my editor")
        self.assertIn("remember", response.lower())
        self.assertEqual(get_agent_memory().long_term.count(), before + 1)

    def test_remember_is_not_mistaken_for_a_coding_task(self):
        route = route_command("remember that the bug is in music_intent")
        self.assertEqual(route["type"], "remember")


class AgentRouteTests(unittest.TestCase):
    def test_an_explicit_agent_prefix_routes_to_the_agent(self):
        for command, goal in (
            ("agent: organize my downloads folder", "organize my downloads folder"),
            ("jarvis, figure out why the tests fail", "figure out why the tests fail"),
            ("jarvis, work on the voice bug", "work on the voice bug"),
        ):
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["type"], "agent_task")
                self.assertEqual(route["goal"], goal)

    def test_the_verb_is_kept_in_the_goal(self):
        # "why the tests fail" alone reads as a question, not a task.
        self.assertTrue(route_command("jarvis, figure out why the tests fail")["goal"].startswith("figure out"))


class EscalationTests(unittest.TestCase):
    def tearDown(self):
        reload_config()

    def test_escalation_is_unavailable_without_a_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            reload_config()
            from providers.registry import register_provider
            from providers.anthropic_provider import AnthropicProvider

            register_provider("anthropic", AnthropicProvider)
            self.assertFalse(agent_module._agent_escalation_available())

    def test_escalation_can_be_switched_off_by_configuration(self):
        with patch.dict(os.environ, {"JARVIS_AGENT_ESCALATION": "false"}):
            reload_config()
            self.assertFalse(agent_module._agent_escalation_available())

    def test_an_unresolvable_request_escalates_when_a_provider_exists(self):
        with patch.object(agent_module, "_agent_escalation_available", return_value=True):
            with patch.object(agent_module, "_run_agent_with_loop", return_value="handled by the agent") as escalate:
                response = agent_module.run_agent(
                    "look through this project and explain how it all fits together",
                    route={"type": "ai", "message": "look through this project and explain how it all fits together"},
                )
        self.assertEqual(response, "handled by the agent")
        escalate.assert_called_once()

    def test_without_a_provider_the_same_request_uses_the_pre_existing_path(self):
        with patch.object(agent_module, "_agent_escalation_available", return_value=False):
            with patch.object(agent_module, "_run_agent_with_loop") as escalate:
                with patch.object(agent_module, "ask_ai", return_value="a plain answer"):
                    goal = "look through this project and explain how it all fits together"
                    response = agent_module.run_agent(goal, route={"type": "ai", "message": goal})
        escalate.assert_not_called()
        self.assertEqual(response, "a plain answer")

    def test_the_agent_route_reports_honestly_with_no_provider(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            reload_config()
            response = agent_module.run_agent("agent: do something ambitious")
        self.assertIn("ANTHROPIC_API_KEY", response)


class VoiceDispatchTests(unittest.TestCase):
    def test_agent_routes_are_recognized_for_off_thread_dispatch(self):
        from voice.background_assistant import AlwaysOnAssistant

        self.assertTrue(AlwaysOnAssistant._is_agent_route({"type": "agent_task", "goal": "x"}))
        self.assertFalse(AlwaysOnAssistant._is_agent_route({"type": "tool", "tool": "open_website"}))

    def test_plan_routes_only_dispatch_off_thread_when_a_provider_exists(self):
        from voice.background_assistant import AlwaysOnAssistant

        with patch.object(agent_module, "_agent_escalation_available", return_value=False):
            self.assertFalse(AlwaysOnAssistant._is_agent_route({"type": "plan", "message": "x"}))
        with patch.object(agent_module, "_agent_escalation_available", return_value=True):
            self.assertTrue(AlwaysOnAssistant._is_agent_route({"type": "plan", "message": "x"}))

    def test_the_agent_answer_is_spoken_rather_than_flattened(self):
        from voice.response_formatter import compose_contextual_ack, format_spoken_response

        self.assertEqual(compose_contextual_ack({"type": "agent_task"}), "I'll work on that, sir.")
        spoken = format_spoken_response(
            "agent: organize downloads",
            {"type": "agent_task"},
            "I moved twelve files into three folders, sir.",
            lang="en",
            execution={"success": True},
        )
        self.assertEqual(spoken, "I moved twelve files into three folders, sir.")

    def test_urls_are_still_stripped_from_agent_speech(self):
        from voice.response_formatter import format_spoken_response

        spoken = format_spoken_response(
            "agent: x", {"type": "agent_task"}, "Details at https://example.com/report", lang="en", execution={"success": True}
        )
        self.assertNotIn("https://", spoken)


if __name__ == "__main__":
    unittest.main()


class TaskControlTests(unittest.TestCase):
    """"What are you doing?" and "cancel" must reach the agent task manager,
    not only the pre-existing read-only scheduler."""

    def setUp(self):
        import threading

        from tasks.manager import get_task_manager

        self.manager = get_task_manager()
        self.release = threading.Event()
        self.handle = self.manager.submit("research the roman empire", lambda task: self.release.wait(10))
        time.sleep(0.1)

    def tearDown(self):
        self.release.set()
        self.handle.wait(timeout=5)

    def test_status_reports_a_running_agent_task(self):
        self.assertIn("research the roman empire", agent_module.run_agent("what are you doing"))

    def test_cancel_stops_a_running_agent_task(self):
        self.assertEqual(agent_module.run_agent("cancel"), "Cancelled.")
        self.assertTrue(self.handle.task.cancelled)

    def test_status_lookup_never_raises(self):
        with patch("tasks.manager.get_task_manager", side_effect=RuntimeError("boom")):
            self.assertEqual(agent_module._active_agent_tasks(), [])
            self.assertEqual(agent_module._cancel_agent_tasks(), 0)
