"""End-to-end integration for complex natural-language commands.

Drives the REAL router and the REAL planner, then executes the resulting plan
through the REAL `AgentRuntime` scheduler with only the tool boundary mocked.
No desktop is touched and no paid API is called.

These assert behaviour that generalises -- "both coordinated targets are
planned", "an ordered clause stays ordered" -- rather than pinning the exact
sentences, which are only illustrations of the classes of request.
"""
from __future__ import annotations

import threading
import time
import unittest

from brain.agent_runtime import AgentRuntime
from brain.execution_graph import build_waves
from brain.models import Action, Plan, ToolResult
from brain.router import route_command
from brain.session_context import SessionContext
from brain.task_planner import create_task_plan


class _MockToolRuntime(AgentRuntime):
    """Records what ran, in what order, on which thread. Runs no real tool."""

    def __init__(self, results=None, delay=0.05):
        super().__init__(trace=False)
        self.order: list[str] = []
        self.threads: set[str] = set()
        self._results = results or {}
        self._delay = delay
        self._lock = threading.Lock()

    def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
        time.sleep(self._delay)
        label = f"{action.tool}:{action.args.get('app_name') or action.args.get('text') or action.args.get('command') or ''}"
        with self._lock:
            self.order.append(label)
            self.threads.add(threading.current_thread().name)
        if action.tool in self._results:
            return self._results[action.tool]
        return ToolResult(True, action.tool, f"{action.tool} ok", data={"summary": f"{action.tool} ok"})


def _plan_for(command: str) -> Plan:
    plan = create_task_plan(command)
    assert plan is not None, f"no plan produced for {command!r}"
    return plan


class IndependentActionsIntegrationTests(unittest.TestCase):
    def test_two_coordinated_launches_are_planned_and_overlap(self):
        plan = _plan_for("open chrome and spotify")
        launched = [a.args["app_name"] for a in plan.actions if a.tool == "open_application"]
        self.assertEqual(len(launched), 2, f"both targets must be planned, got {launched}")

        runtime = _MockToolRuntime(delay=0.2)
        started = time.perf_counter()
        results = runtime.execute(plan)
        elapsed = time.perf_counter() - started

        self.assertTrue(all(r.success for r in results))
        self.assertLess(elapsed, 0.75, "independent launches must overlap")
        self.assertGreater(len(runtime.threads), 1, "parallel work must actually use more than one thread")

    def test_three_coordinated_launches_all_run(self):
        plan = _plan_for("open chrome, spotify and notepad")
        launched = sorted(a.args["app_name"] for a in plan.actions if a.tool == "open_application")
        self.assertEqual(len(launched), 3, f"got {launched}")


class OrderedActionsIntegrationTests(unittest.TestCase):
    def test_a_clause_needing_the_window_stays_after_it(self):
        plan = _plan_for("open notepad and type hello")
        runtime = _MockToolRuntime()
        runtime.execute(plan)
        self.assertEqual(runtime.order[0], "open_application:notepad")
        self.assertEqual(runtime.order[-1], "type_text:hello")
        self.assertEqual(runtime.threads, {"MainThread"}, "an ordered plan must not be parallelised")

    def test_an_explicit_then_is_honoured(self):
        plan = _plan_for("open notepad, then type hello")
        self.assertEqual(build_waves(plan.actions), [[0], [1], [2]])


class ResultPassingIntegrationTests(unittest.TestCase):
    def test_a_later_action_writes_what_an_earlier_action_produced(self):
        # "run the tests and write the failures down" as a structured plan.
        plan = Plan("run the tests and write down the failures", [
            Action("run_command", {"command": "pytest"}),
            Action("write_text_file", {
                "path": "failures.txt",
                "contents": {"__from_result__": {"action": 0, "field": "data.failures"}},
            }, depends_on=[0]),
        ])
        captured = {}

        class Capturing(_MockToolRuntime):
            def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
                if action.tool == "run_command":
                    return ToolResult(True, "run_command", "2 failed", data={"failures": ["a::one", "b::two"], "summary": "2 failed"})
                captured.update(action.args)
                return ToolResult(True, action.tool, "written")

        results = Capturing().execute(plan)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(captured["contents"], "a::one\nb::two")


class FailureIsolationIntegrationTests(unittest.TestCase):
    def test_one_launch_failing_does_not_hide_the_other_succeeding(self):
        plan = _plan_for("open chrome and spotify")

        class PartlyFailing(_MockToolRuntime):
            def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
                if action.args.get("app_name") == "spotify":
                    return ToolResult(False, action.tool, "could not start", error="application_not_found")
                return ToolResult(True, action.tool, "ok")

        results = PartlyFailing().execute(plan)
        self.assertTrue(any(r.success for r in results), "the healthy launch must still report success")
        self.assertTrue(any(not r.success for r in results), "the real failure must still be reported")


class RoutingStaysFastTests(unittest.TestCase):
    def test_a_trivial_command_makes_no_model_call_and_routes_quickly(self):
        started = time.perf_counter()
        route = route_command("open notepad", context=SessionContext())
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route.get("model_calls", 0), 0)
        self.assertLess(elapsed_ms, 250, "the deterministic fast path must stay fast")


if __name__ == "__main__":
    unittest.main()
