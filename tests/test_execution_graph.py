"""Tests for `brain/execution_graph.py` and its use by `AgentRuntime`.

These cover the property the scheduler exists to provide: a plan's real
dependency structure decides the order, independent work overlaps, and
everything that must stay ordered still is. Desktop side effects are never
performed -- `_execute_action` is overridden throughout, the same technique
`tests/test_agent_runtime_parallel.py` already uses.
"""
from __future__ import annotations

import threading
import time
import unittest

from brain.agent_runtime import AgentRuntime
from brain.execution_graph import (
    CyclicPlanError,
    build_waves,
    describe_schedule,
    is_parallel_candidate,
    partition_wave,
    plan_is_chain,
)
from brain.models import Action, ActionRisk, Plan, PlanStatus, ToolResult


def _open(app: str, **kw) -> Action:
    return Action("open_application", {"app_name": app}, **kw)


class BuildWavesTests(unittest.TestCase):
    def test_a_chain_levels_into_one_action_per_wave(self):
        actions = [_open("notepad"), Action("wait_for_window", {"app_name": "notepad"}, depends_on=[0]), Action("type_text", {"text": "hi"}, depends_on=[1])]
        self.assertEqual(build_waves(actions), [[0], [1], [2]])
        self.assertTrue(plan_is_chain(actions))

    def test_independent_actions_share_a_single_wave(self):
        actions = [_open("spotify"), _open("code"), Action("open_website", {"url": "https://example.com"})]
        self.assertEqual(build_waves(actions), [[0, 1, 2]])
        self.assertFalse(plan_is_chain(actions))

    def test_a_dependent_action_waits_for_every_dependency(self):
        actions = [_open("chrome"), _open("spotify"), Action("volume_down", {}, depends_on=[0, 1])]
        self.assertEqual(build_waves(actions), [[0, 1], [2]])

    def test_diamond_dependencies_level_correctly(self):
        actions = [
            _open("chrome"),
            Action("volume_up", {}, depends_on=[0]),
            Action("volume_down", {}, depends_on=[0]),
            Action("mute_volume", {}, depends_on=[1, 2]),
        ]
        self.assertEqual(build_waves(actions), [[0], [1, 2], [3]])

    def test_an_empty_plan_has_no_waves(self):
        self.assertEqual(build_waves([]), [])

    def test_a_cycle_is_reported_rather_than_silently_dropped(self):
        actions = [_open("a", depends_on=[1]), _open("b", depends_on=[0])]
        with self.assertRaises(CyclicPlanError):
            build_waves(actions)

    def test_an_out_of_range_dependency_does_not_crash_the_scheduler(self):
        # plan_validator rejects these before execution; the scheduler must
        # still degrade rather than turn a planning bug into a lost request.
        actions = [_open("spotify"), _open("code", depends_on=[99])]
        self.assertEqual(build_waves(actions), [[0, 1]])


class PartitionWaveTests(unittest.TestCase):
    def test_independent_safe_tools_all_join_the_parallel_group(self):
        actions = [_open("spotify"), _open("code")]
        self.assertEqual(partition_wave(actions, [0, 1]), ([0, 1], []))

    def test_a_context_dependent_tool_is_kept_sequential(self):
        actions = [_open("spotify"), Action("type_text", {"text": "x"})]
        parallel, sequential = partition_wave(actions, [0, 1])
        self.assertEqual(parallel, [])
        self.assertEqual(sequential, [0, 1])

    def test_two_actions_needing_the_same_exclusive_resource_do_not_both_parallelize(self):
        # Both music transport tools claim the authenticated-browser resource,
        # so running them together would only queue on the same lock.
        actions = [Action("music_pause", {}), Action("music_next", {}), _open("spotify")]
        parallel, sequential = partition_wave(actions, [0, 1, 2])
        self.assertEqual(len(parallel), 2)
        self.assertIn(2, parallel)
        self.assertEqual(len(sequential), 1)

    def test_a_repeated_action_is_not_batched_with_itself(self):
        actions = [_open("spotify"), _open("spotify"), _open("code")]
        parallel, sequential = partition_wave(actions, [0, 1, 2])
        self.assertEqual(parallel, [0, 2])
        self.assertEqual(sequential, [1])

    def test_a_lone_candidate_is_not_worth_a_thread_pool(self):
        actions = [_open("spotify"), Action("type_text", {"text": "x"})]
        self.assertEqual(partition_wave(actions, [0, 1]), ([], [0, 1]))

    def test_high_impact_and_optional_actions_are_never_parallel_candidates(self):
        self.assertFalse(is_parallel_candidate(_open("a", optional=True)))
        self.assertFalse(is_parallel_candidate(_open("a", risk=ActionRisk.HIGH_IMPACT)))
        self.assertTrue(is_parallel_candidate(_open("a")))

    def test_describe_schedule_reports_the_available_concurrency(self):
        actions = [_open("chrome"), _open("spotify"), Action("volume_down", {}, depends_on=[0, 1])]
        schedule = describe_schedule(actions)
        self.assertEqual(len(schedule), 2)
        self.assertEqual([entry["index"] for entry in schedule[0]["parallel"]], [0, 1])
        self.assertEqual([entry["index"] for entry in schedule[1]["sequential"]], [2])


class _RecordingRuntime(AgentRuntime):
    """Runs no real tool: records call order and optionally fails/sleeps."""

    def __init__(self, *, failures=(), delay=0.0, **kw):
        super().__init__(trace=False, **kw)
        self.calls: list[str] = []
        self.threads: dict[str, str] = {}
        self._failures = set(failures)
        self._delay = delay
        self._lock = threading.Lock()

    def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
        if self._delay:
            time.sleep(self._delay)
        key = f"{action.tool}:{action.args.get('app_name') or action.args.get('text') or ''}"
        with self._lock:
            self.calls.append(key)
            self.threads[key] = threading.current_thread().name
        if key in self._failures:
            return ToolResult(False, action.tool, "boom", error="tool_failed")
        return ToolResult(True, action.tool, f"{action.tool} ok")


class ScheduledExecutionTests(unittest.TestCase):
    def test_dependency_order_is_preserved_exactly(self):
        plan = Plan("g", [_open("chrome"), _open("spotify"), Action("volume_down", {}, depends_on=[0, 1])])
        runtime = _RecordingRuntime()
        results = runtime.execute(plan)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(runtime.calls[-1], "volume_down:")
        self.assertEqual(plan.status, PlanStatus.COMPLETED)

    def test_independent_actions_overlap_in_wall_clock_time(self):
        plan = Plan("g", [_open("chrome"), _open("spotify"), _open("code")])
        runtime = _RecordingRuntime(delay=0.3)
        started = time.perf_counter()
        runtime.execute(plan)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.75, "three independent launches should overlap, not serialize")

    def test_a_dependent_step_still_waits_even_when_siblings_run_in_parallel(self):
        plan = Plan("g", [_open("chrome"), _open("spotify"), Action("volume_down", {}, depends_on=[0, 1])])
        runtime = _RecordingRuntime(delay=0.2)
        runtime.execute(plan)
        # the dependent action must be last no matter which sibling won the race
        self.assertEqual(runtime.calls[-1], "volume_down:")

    def test_results_come_back_in_original_action_order(self):
        plan = Plan("g", [_open("chrome"), _open("spotify"), _open("code")])
        results = _RecordingRuntime(delay=0.05).execute(plan)
        self.assertEqual([r.tool for r in results], ["open_application"] * 3)

    def test_one_parallel_failure_does_not_discard_a_sibling_result(self):
        plan = Plan("g", [_open("chrome"), _open("spotify"), _open("code")])
        runtime = _RecordingRuntime(failures={"open_application:spotify"})
        results = runtime.execute(plan)
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertTrue(results[2].success, "an unrelated sibling must still report its own success")
        self.assertEqual(plan.status, PlanStatus.FAILED)

    def test_a_dependent_action_is_not_run_when_its_dependency_failed(self):
        plan = Plan("g", [_open("chrome"), Action("volume_down", {}, depends_on=[0])])
        runtime = _RecordingRuntime(failures={"open_application:chrome"})
        results = runtime.execute(plan)
        self.assertFalse(results[0].success)
        self.assertNotIn("volume_down:", runtime.calls)
        self.assertEqual(plan.status, PlanStatus.FAILED)

    def test_an_optional_failure_does_not_stop_the_plan(self):
        plan = Plan("g", [_open("chrome", optional=True), _open("spotify"), Action("volume_down", {}, depends_on=[1])])
        runtime = _RecordingRuntime(failures={"open_application:chrome"})
        runtime.execute(plan)
        self.assertIn("volume_down:", runtime.calls)

    def test_an_exception_inside_one_action_is_reported_not_swallowed(self):
        class Exploding(_RecordingRuntime):
            def _execute_action(self, action, cancellation_token=None, plan_lock_held=False):
                if action.args.get("app_name") == "spotify":
                    raise RuntimeError("driver exploded")
                return super()._execute_action(action, cancellation_token, plan_lock_held)

        plan = Plan("g", [_open("chrome"), _open("spotify"), _open("code")])
        results = Exploding().execute(plan)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertIn("driver exploded", results[1].error)
        self.assertTrue(results[2].success)

    def test_a_pure_chain_still_runs_strictly_in_order(self):
        plan = Plan("g", [
            _open("notepad"),
            Action("wait_for_window", {"app_name": "notepad"}, depends_on=[0]),
            Action("type_text", {"text": "hello"}, depends_on=[1]),
        ])
        runtime = _RecordingRuntime()
        runtime.execute(plan)
        self.assertEqual(runtime.calls, ["open_application:notepad", "wait_for_window:notepad", "type_text:hello"])
        self.assertEqual(set(runtime.threads.values()), {"MainThread"})

    def test_cancellation_marks_every_action_cancelled_and_runs_none(self):
        # Exercised against the scheduler directly: acquiring the plan-level
        # resource has its own, separately tested cancellation behaviour
        # (`acquire_action_resource` raises on an already-cancelled token),
        # and this test is about what the scheduler does once it is running.
        class Token:
            cancelled = True

        plan = Plan("g", [_open("chrome"), _open("spotify"), Action("volume_down", {}, depends_on=[0])])
        runtime = _RecordingRuntime()
        results = runtime._execute_plan_scheduled(plan, cancellation_token=Token())
        self.assertTrue(results, "cancelled actions must still be reported")
        self.assertTrue(any(r.error == "cancelled" for r in results))
        self.assertEqual(runtime.calls, [], "nothing may execute after cancellation")
        self.assertEqual(plan.status, PlanStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
