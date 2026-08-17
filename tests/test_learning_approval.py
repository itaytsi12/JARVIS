import unittest

from voice.learning_approval import (
    DEFAULT_LEARNING_QUESTION,
    LearningApprovalOutcome,
    classify_approval_response,
    normalize_approval_text,
    request_learning_approval,
)


class StepClock:
    def __init__(self, step=1.0):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class NormalizationTests(unittest.TestCase):
    def test_exact_phrase_matches(self):
        self.assertEqual(classify_approval_response("yes jarvis"), LearningApprovalOutcome.APPROVED)
        self.assertEqual(classify_approval_response("no jarvis"), LearningApprovalOutcome.DECLINED)

    def test_case_and_punctuation_are_normalized(self):
        self.assertEqual(classify_approval_response("Yes, Jarvis!"), LearningApprovalOutcome.APPROVED)
        self.assertEqual(classify_approval_response("  NO JARVIS.  "), LearningApprovalOutcome.DECLINED)
        self.assertEqual(classify_approval_response("Yes   Jarvis"), LearningApprovalOutcome.APPROVED)

    def test_bare_yes_is_not_approval(self):
        for text in ("yes", "yeah", "okay", "ok", "sure", "yep"):
            with self.subTest(text=text):
                self.assertIsNone(classify_approval_response(text))

    def test_bare_no_is_not_decline(self):
        for text in ("no", "nope", "nah"):
            with self.subTest(text=text):
                self.assertIsNone(classify_approval_response(text))

    def test_unrelated_speech_is_not_an_answer(self):
        self.assertIsNone(classify_approval_response("open notepad"))
        self.assertIsNone(classify_approval_response(""))
        self.assertIsNone(classify_approval_response(None))

    def test_jarvis_alone_is_not_an_answer(self):
        self.assertIsNone(classify_approval_response("jarvis"))

    def test_normalize_collapses_whitespace(self):
        self.assertEqual(normalize_approval_text("yes    jarvis"), "yes jarvis")


class RequestLearningApprovalTests(unittest.TestCase):
    def test_speaks_the_exact_question_by_default(self):
        spoken = []
        request_learning_approval(speak_fn=spoken.append, listen_fn=lambda t: "yes jarvis", clock=StepClock())
        self.assertEqual(spoken, [DEFAULT_LEARNING_QUESTION])

    def test_yes_jarvis_approves_immediately(self):
        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=lambda t: "yes jarvis", clock=StepClock())
        self.assertEqual(result.outcome, LearningApprovalOutcome.APPROVED)
        self.assertEqual(result.transcript, "yes jarvis")

    def test_no_jarvis_declines_immediately(self):
        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=lambda t: "no jarvis", clock=StepClock())
        self.assertEqual(result.outcome, LearningApprovalOutcome.DECLINED)

    def test_bare_yes_does_not_approve(self):
        calls = {"n": 0}

        def listen(remaining):
            calls["n"] += 1
            return "yes" if calls["n"] == 1 else None

        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=listen, clock=StepClock(step=5.0), timeout_seconds=30)
        self.assertEqual(result.outcome, LearningApprovalOutcome.TIMED_OUT)

    def test_no_valid_answer_within_window_times_out(self):
        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=lambda t: None, clock=StepClock(step=5.0), timeout_seconds=30)
        self.assertEqual(result.outcome, LearningApprovalOutcome.TIMED_OUT)

    def test_timeout_means_no_and_never_hangs(self):
        clock = StepClock(step=31.0)
        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=lambda t: None, clock=clock, timeout_seconds=30)
        self.assertEqual(result.outcome, LearningApprovalOutcome.TIMED_OUT)

    def test_unrecognized_speech_keeps_the_window_open(self):
        calls = {"n": 0}

        def listen(remaining):
            calls["n"] += 1
            return "what's the weather" if calls["n"] == 1 else "yes jarvis"

        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=listen, clock=StepClock(step=2.0), timeout_seconds=30)
        self.assertEqual(result.outcome, LearningApprovalOutcome.APPROVED)
        self.assertEqual(calls["n"], 2)

    def test_cancellation_token_stops_the_wait(self):
        class Token:
            cancelled = True

        result = request_learning_approval(speak_fn=lambda t: None, listen_fn=lambda t: None, clock=StepClock(), cancellation_token=Token())
        self.assertEqual(result.outcome, LearningApprovalOutcome.CANCELLED)


if __name__ == "__main__":
    unittest.main()
