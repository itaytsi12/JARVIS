"""End-to-end verification of the whole agent path, without Claude.

This is the integration test the project brief asks for: a scripted
provider drives real task creation, real tool execution, real
observations, a real second action, real completion, a real memory write
and a real episode -- so the architecture can be validated offline and
for free.

The real-Claude smoke test is `scripts/test_claude_agent.py`, which only
runs when an API key is explicitly provided.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.agent_service import run_agent_task, submit_agent_task
from config.settings import reload_config
from memory.agent_memory import AgentMemory
from memory.agent_store import AgentDatabase
from providers.base import ModelResponse, ToolCall, Usage
from providers.mock_provider import CallableProvider, ScriptedProvider, text_response, tool_response
from providers.usage import UsageStore
from tasks.manager import TaskManager
from tasks.models import TaskCancelled, TaskKind


def _memory() -> AgentMemory:
    return AgentMemory(AgentDatabase(Path(tempfile.mkdtemp()) / "agent.sqlite3"))


class CodingWorkflowTests(unittest.TestCase):
    """The headline scenario: run it, see it break, fix it, prove the fix."""

    def setUp(self):
        self.project = Path(tempfile.mkdtemp())
        (self.project / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (self.project / "test_calc.py").write_text(
            "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
        )
        self.memory = _memory()

    def _provider(self):
        return ScriptedProvider(
            [
                tool_response("run_command", {"command": "python -m pytest -q", "working_directory": str(self.project)}, call_id="c1"),
                tool_response("read_code", {"path": str(self.project / "calc.py")}, call_id="c2"),
                tool_response(
                    "edit_code",
                    {"path": str(self.project / "calc.py"), "old_text": "return a - b", "new_text": "return a + b"},
                    call_id="c3",
                ),
                tool_response("run_command", {"command": "python -m pytest -q", "working_directory": str(self.project)}, call_id="c4"),
                text_response("The failing test passes now, sir."),
            ]
        )

    def test_the_agent_reproduces_fixes_and_verifies(self):
        outcome = run_agent_task("run the tests in my project and fix what breaks", provider=self._provider(), memory=self.memory)
        self.assertTrue(outcome.success)
        self.assertTrue(outcome.verified)
        self.assertEqual(outcome.stop_reason, "completed")
        self.assertEqual(outcome.answer, "The failing test passes now, sir.")

    def test_the_fix_is_real_and_the_evidence_is_recorded(self):
        outcome = run_agent_task("run the tests in my project and fix what breaks", provider=self._provider(), memory=self.memory)
        self.assertIn("return a + b", (self.project / "calc.py").read_text(encoding="utf-8"))
        steps = outcome.run.steps
        self.assertEqual([step.tool for step in steps], ["run_command", "read_code", "edit_code", "run_command"])
        # The first run genuinely failed and the last genuinely passed --
        # the agent did not simply declare victory after an edit.
        self.assertFalse(steps[0].success)
        self.assertTrue(steps[-1].success)

    def test_a_complete_episode_is_stored_for_training(self):
        outcome = run_agent_task("run the tests in my project and fix what breaks", provider=self._provider(), memory=self.memory)
        episode = self.memory.episodes.get(outcome.episode_id)
        self.assertIsNotNone(episode)
        self.assertEqual(episode.route, "agent")
        self.assertEqual(episode.step_count, 4)
        self.assertTrue(episode.success)
        self.assertIn("input_tokens", episode.token_usage)
        self.assertIn("coding", episode.plan)
        self.assertTrue(episode.context_summary)

    def test_the_raw_jsonl_dataset_is_appended(self):
        run_agent_task("run the tests in my project and fix what breaks", provider=self._provider(), memory=self.memory)
        path = Path(self.memory.episodes.jsonl_path)
        self.assertTrue(path.exists())
        self.assertTrue(path.read_text(encoding="utf-8").strip())

    def test_conversation_history_records_both_sides(self):
        run_agent_task("run the tests in my project and fix what breaks", provider=self._provider(), memory=self.memory)
        turns = self.memory.conversation.recent_turns(self.memory.session_id, 10)
        self.assertEqual([turn.role for turn in turns], ["user", "assistant"])


class MemoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.memory = _memory()

    def test_an_explicit_memory_is_written_and_later_retrieved(self):
        provider = ScriptedProvider([text_response("Noted, sir.")])
        outcome = run_agent_task(
            "Remember that my main Jarvis project is at C:/Users/Ori/Desktop/jarvis", provider=provider, memory=self.memory
        )
        self.assertTrue(self.memory.long_term.count())
        episode = self.memory.episodes.get(outcome.episode_id)
        self.assertTrue(episode.memories_written)

        # A later, differently-worded request retrieves it.
        retrieved = self.memory.retrieve("open my main jarvis project")
        self.assertTrue(any("jarvis" in item.memory.text.lower() for item in retrieved.memories))

    def test_an_ordinary_command_writes_no_long_term_memory(self):
        provider = ScriptedProvider([tool_response("get_time"), text_response("It's half past one, sir.")])
        run_agent_task("what time is it", provider=provider, memory=self.memory)
        self.assertEqual(self.memory.long_term.count(), 0)

    def test_the_agent_can_store_and_recall_through_its_tools(self):
        store = ScriptedProvider(
            [tool_response("remember_fact", {"text": "the user's editor is vscode"}), text_response("Noted, sir.")]
        )
        run_agent_task("note my editor", provider=store, memory=self.memory)
        self.assertTrue(self.memory.long_term.count())

        recall = ScriptedProvider([tool_response("recall_memory", {"query": "editor"}), text_response("You use vscode, sir.")])
        outcome = run_agent_task("which editor do I use", provider=recall, memory=self.memory)
        self.assertIn("vscode", outcome.run.steps[0].observation)


class NoProviderTests(unittest.TestCase):
    def test_without_a_key_the_agent_says_so_and_still_records_an_episode(self):
        memory = _memory()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            reload_config()
            outcome = run_agent_task("do something complicated", memory=memory)
        reload_config()
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.stop_reason, "no_provider")
        self.assertIn("ANTHROPIC_API_KEY", outcome.answer)
        self.assertIsNotNone(memory.episodes.get(outcome.episode_id))
        self.assertEqual(outcome.run.model_calls, 0)


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager(persist=False)

    def tearDown(self):
        self.manager.shutdown(wait=False)

    def test_two_independent_agent_tasks_run_side_by_side(self):
        first = submit_agent_task(
            "research the roman empire",
            manager=self.manager,
            provider=ScriptedProvider([text_response("Rome fell in 476, sir.")]),
            memory=_memory(),
        )
        second = submit_agent_task(
            "run the tests in my project",
            manager=self.manager,
            provider=ScriptedProvider([text_response("All tests pass, sir.")]),
            memory=_memory(),
        )
        self.assertEqual(first.result(timeout=20), "Rome fell in 476, sir.")
        self.assertEqual(second.result(timeout=20), "All tests pass, sir.")

    def test_a_desktop_goal_is_scheduled_as_ui_exclusive(self):
        handle = submit_agent_task(
            "open spotify and turn the volume up",
            manager=self.manager,
            provider=ScriptedProvider([text_response("Done, sir.")]),
            memory=_memory(),
        )
        self.assertIs(handle.task.kind, TaskKind.EXCLUSIVE_UI)
        handle.wait(timeout=20)

    def test_a_research_goal_is_scheduled_as_concurrent(self):
        handle = submit_agent_task(
            "look up what the roman empire traded",
            manager=self.manager,
            provider=ScriptedProvider([text_response("Mostly grain, sir.")]),
            memory=_memory(),
        )
        self.assertIs(handle.task.kind, TaskKind.CONCURRENT)
        handle.wait(timeout=20)

    def test_a_running_agent_task_can_be_cancelled(self):
        import threading

        entered = threading.Event()

        def handler(messages, **kwargs):
            entered.set()
            # Keep asking for another tool call so the loop keeps going
            # until the cancellation token is noticed.
            return ModelResponse(
                text="", tool_calls=[ToolCall("c", "calculator", {"expression": "1+1"})], stop_reason="tool_use", usage=Usage(1, 1)
            )

        handle = submit_agent_task(
            "keep calculating forever",
            manager=self.manager,
            provider=CallableProvider(handler),
            memory=_memory(),
        )
        self.assertTrue(entered.wait(10))
        self.assertTrue(handle.cancel())
        try:
            handle.result(timeout=20)
        except TaskCancelled:
            pass
        self.assertTrue(handle.task.cancelled)

    def test_task_observations_are_recorded_as_the_agent_works(self):
        handle = submit_agent_task(
            "add two numbers",
            manager=self.manager,
            provider=ScriptedProvider([tool_response("calculator", {"expression": "1+1"}), text_response("Two, sir.")]),
            memory=_memory(),
        )
        handle.result(timeout=20)
        self.assertTrue(handle.task.observations)
        self.assertEqual(handle.task.observations[0].source, "calculator")


class CostTrackingTests(unittest.TestCase):
    def test_model_usage_is_persisted_per_task(self):
        store = UsageStore(Path(tempfile.mkdtemp()) / "usage.sqlite3")
        with patch("providers.usage.get_usage_store", return_value=store):
            outcome = run_agent_task(
                "what time is it",
                provider=ScriptedProvider([tool_response("get_time"), text_response("Half one, sir.")]),
                memory=_memory(),
            )
        summary = store.for_task(outcome.task_id)
        self.assertEqual(summary.calls, 2)
        self.assertGreater(summary.total_tokens, 0)
        store.close()


if __name__ == "__main__":
    unittest.main()
