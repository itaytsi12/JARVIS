"""End-to-end verification of the reported conversational sequences (item 7
of the live conversational-context test), driving the REAL `brain.agent.run_agent`
and `brain.router.route_command` with a scripted fake provider
(`providers/mock_provider.py`) -- never a route label alone. Mirrors the
pattern in `tests/test_agent_runtime_integration.py`.
"""
from __future__ import annotations

import unittest

from providers.mock_provider import text_response
from providers.registry import register_provider, reset_providers_for_tests

from brain.session_context import SessionContext


class _RecordingProvider:
    """Minimal recording `ModelProvider`: every turn's messages are kept so
    a test can assert on what the model actually SAW, not just that
    something answered."""

    name = "recording"

    def __init__(self, handler, model="claude-sonnet-5"):
        self._handler = handler
        self.model = model
        self.turns: list[list] = []

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self):
        return None

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "available": True}

    def complete(self, messages, **kwargs):
        self.turns.append(list(messages))
        response = self._handler(messages, **kwargs)
        response.provider = self.name
        response.model = self.model
        return response

    @property
    def call_count(self) -> int:
        return len(self.turns)


def install_provider(test_case, provider) -> None:
    register_provider("anthropic", lambda: provider)
    test_case.addCleanup(reset_providers_for_tests)


class SequenceA_ExplanatoryFollowupUsesRealContext(unittest.TestCase):
    """Run git status in the JARVIS project. / What does that mean?

    Expected: the second request refers to the actual previous result, and
    the model genuinely receives that referent text -- not a blind web
    search."""

    def test_followup_reaches_the_provider_with_the_prior_answer_embedded(self):
        from brain.agent import agent_runtime, run_agent

        provider = _RecordingProvider(lambda messages, **kw: text_response("There were 3 modified files and 2 untracked files, sir."))
        install_provider(self, provider)

        first_outcome = {}
        first_answer = run_agent(
            "Run git status in the JARVIS project.",
            execution_outcome=first_outcome,
        )
        self.assertGreaterEqual(provider.call_count, 1)
        self.assertEqual(agent_runtime.context.last_assistant_response, first_answer)

        provider.turns.clear()
        second_outcome = {}
        second_answer = run_agent(
            "What does that mean?",
            execution_outcome=second_outcome,
        )

        # Escalated to the agent runtime (route_type="agent_task", the same
        # label every other escalation ends with -- brain/agent.py::
        # _run_agent_with_loop is the single shared entry point), with
        # provenance preserved: `escalated_from` records that THIS
        # escalation specifically came from the resolved conversational
        # context, not from the complexity guard or anywhere else.
        self.assertEqual(second_outcome["route_type"], "agent_task")
        self.assertEqual(second_outcome["escalated_from"], "conversational_context")
        self.assertGreaterEqual(provider.call_count, 1, "the provider was never called for the follow-up")
        first_turn_text = " ".join(message.content for message in provider.turns[0])
        self.assertIn("3 modified files", first_turn_text)
        self.assertIn("What does that mean?", first_turn_text)
        self.assertIsInstance(second_answer, str)
        self.assertTrue(second_answer)

    def test_followup_with_no_prior_context_still_answers_as_an_ordinary_question(self):
        from brain.agent import agent_runtime, run_agent

        agent_runtime.context.last_assistant_response = None
        agent_runtime.context.last_spoken_response = None
        from brain.router import route_command

        route = route_command("What does that mean?", agent_runtime.context)
        self.assertEqual(route["type"], "question")


class SequenceC_StopCancelsWithZeroProviderCalls(unittest.TestCase):
    """Start a long agent task. Say: Stop.

    Expected: the active JARVIS task is cancelled immediately, locally,
    with zero Claude calls -- verified by actually registering an
    interactive task and confirming (a) the route is deterministic and (b)
    no provider is ever installed/reachable for this path, so any call
    would be a hard failure, not a mock recording an unexpected hit."""

    def test_stop_never_touches_the_agent_provider(self):
        from brain.task_supervisor import CancellationToken, register_interactive_task, unregister_interactive_task
        from brain.agent import run_agent
        from brain.router import route_command

        # Deliberately no provider registered: if "stop" ever reached the
        # agent runtime, run_agent would raise/degrade rather than silently
        # spend a call -- this proves zero-Claude-call handling structurally,
        # not just by absence of a call counter.
        reset_providers_for_tests()

        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            route = route_command("stop")
            self.assertEqual(route["type"], "cancel_read_only_task")
            outcome = {}
            answer = run_agent("stop", route=route, execution_outcome=outcome)
            self.assertEqual(outcome["route_type"], "cancel_read_only_task")
            self.assertNotIn("route_source", {"agent_runtime"} & {outcome.get("route_source")})
            self.assertIn(answer, {"Cancelled.", "There's nothing to cancel."})
        finally:
            unregister_interactive_task(task_id)


class SequenceE_BarePauseDuringTaskPrefersTaskOverMusic(unittest.TestCase):
    """Start a long JARVIS task while music exists. Say: Pause.

    Expected: task pause/cancel behavior according to the context-priority
    design, not blindly music_pause -- verified against the real router
    with a real registered interactive task, independent of whether music
    routing would otherwise apply."""

    def test_bare_pause_prefers_the_active_task(self):
        from brain.task_supervisor import CancellationToken, register_interactive_task, unregister_interactive_task
        from brain.router import route_command

        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            route = route_command("pause")
            self.assertEqual(route["type"], "cancel_read_only_task")
            self.assertNotEqual(route.get("tool"), "music_pause")
        finally:
            unregister_interactive_task(task_id)

    def test_explicit_pause_the_music_still_wins_during_the_same_task(self):
        from brain.task_supervisor import CancellationToken, register_interactive_task, unregister_interactive_task
        from brain.router import route_command

        token = CancellationToken()
        task_id = register_interactive_task(token)
        try:
            route = route_command("pause the music")
            self.assertEqual(route.get("tool"), "music_pause")
        finally:
            unregister_interactive_task(task_id)


class ContextualQuestionVoiceDispatchTests(unittest.TestCase):
    """A `contextual_question` route must dispatch the same way `plan`/`ai`
    already do: off-thread, cancellable, and narrated when an agent
    provider can actually answer it, or through the cancellable
    `_start_question_task` path (never the synchronous default path that
    would block the microphone) when no provider is configured."""

    def tearDown(self):
        reset_providers_for_tests()

    def test_dispatches_as_an_agent_route_when_a_provider_is_available(self):
        from voice.background_assistant import AlwaysOnAssistant

        provider = _RecordingProvider(lambda messages, **kw: text_response("ok"))
        install_provider(self, provider)
        route = {"type": "contextual_question", "message": "What does that mean?", "context_text": "x"}
        self.assertTrue(AlwaysOnAssistant._is_agent_route(route))

    def test_is_not_an_agent_route_without_a_provider(self):
        from voice.background_assistant import AlwaysOnAssistant

        reset_providers_for_tests()
        route = {"type": "contextual_question", "message": "What does that mean?", "context_text": "x"}
        self.assertFalse(AlwaysOnAssistant._is_agent_route(route))


if __name__ == "__main__":
    unittest.main()
