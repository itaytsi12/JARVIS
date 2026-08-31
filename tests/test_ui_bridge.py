"""The backend -> window channel.

Two halves, deliberately split:

- `ui/model_status.py` needs no Qt at all, so its honesty rules (never
  claim an unconfigured provider, never hard-code a model count) are
  asserted unconditionally.
- `ui/ui_bridge.py` imports PySide6. Those tests skip -- not fail -- when
  it is absent, the same convention `tests/test_hf_backend.py` uses for
  the training venv. They still need no window and no GPU: a QObject and
  a queued signal work under `QCoreApplication`, so the whole event-bus
  mapping is exercised headlessly.
"""
import time
import unittest
from unittest.mock import Mock, patch

from config import events
from ui.model_status import (
    MODEL_IDS,
    STATE_ERROR,
    STATE_RATE_LIMITED,
    VALID_STATES,
    ModelStatus,
    online_caption,
    online_count,
    state_for_error,
)

try:  # PySide6 is optional -- JARVIS runs headless without it.
    from PySide6.QtCore import QCoreApplication

    from ui.ui_bridge import NEUTRAL_EVENT_STATES, UiBridge

    QT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on a headless box
    QT_AVAILABLE = False


class ModelStatusTests(unittest.TestCase):
    def test_the_model_count_is_derived_never_hard_coded(self):
        statuses = [
            ModelStatus("openai", "OpenAI", True),
            ModelStatus("gemini", "Gemini", False, "not_implemented"),
            ModelStatus("anthropic", "Anthropic", True),
        ]
        self.assertEqual(online_count(statuses), 2)
        self.assertEqual(online_caption(2), "2 MODELS ONLINE")
        self.assertEqual(online_caption(1), "1 MODEL ONLINE")
        self.assertEqual(online_caption(0), "0 MODELS ONLINE")

    def test_gemini_is_never_reported_available_because_nothing_implements_it(self):
        from ui.model_status import _gemini_status

        status = _gemini_status()
        self.assertFalse(status.available)
        self.assertEqual(status.reason, "not_implemented")

    def test_a_missing_openai_key_reports_the_reason_rather_than_availability(self):
        from ui.model_status import _openai_status, _vision_status

        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            self.assertEqual(_openai_status().reason, "missing_api_key")
            # Vision rides on the same credential and says so, rather than
            # reporting a different answer about one key.
            self.assertEqual(_vision_status().reason, "missing_openai_api_key")

    def test_a_probe_that_raises_becomes_an_offline_node_not_an_exception(self):
        from ui import model_status

        with patch.dict(model_status._PROBES, {"openai": Mock(side_effect=RuntimeError("x"))}):
            statuses = {status.model_id: status for status in model_status.discover_models()}
        self.assertEqual(len(statuses), len(MODEL_IDS))
        self.assertFalse(statuses["openai"].available)
        self.assertTrue(statuses["openai"].reason.startswith("probe_failed:"))

    def test_a_rate_limit_is_amber_not_red(self):
        """A throttled module is configured and working. Rendering it as a
        red failure would train the user to ignore red."""
        self.assertEqual(state_for_error("RateLimitError"), STATE_RATE_LIMITED)
        self.assertEqual(state_for_error("ProviderRateLimited"), STATE_RATE_LIMITED)
        self.assertEqual(state_for_error("ProviderAuthError"), STATE_ERROR)
        self.assertEqual(state_for_error(""), STATE_ERROR)
        self.assertIn(STATE_RATE_LIMITED, VALID_STATES)

    def test_the_anthropic_provider_publishes_its_translated_error_type(self):
        """The UI keys the amber state on JARVIS's own neutral name, so
        the provider must report that rather than the vendor's."""
        from providers.base import ProviderRateLimited

        self.assertEqual(state_for_error(ProviderRateLimited.__name__), STATE_RATE_LIMITED)


@unittest.skipUnless(QT_AVAILABLE, "PySide6 is not installed in this environment")
class UiBridgeTests(unittest.TestCase):
    """No window is created: `QCoreApplication` plus `processEvents` is
    enough to drive the queued connection the bridge marshals through."""

    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        events.reset_for_tests()
        self.bridge = UiBridge()
        # `start()` is the real subscribe path, so the mapping under test
        # is the one production uses. The availability probe it kicks off
        # is replaced with a fixed, all-available answer -- both to keep
        # the test offline and so a state change is not masked by the
        # "an unconfigured module never lights up" rule.
        statuses = [ModelStatus(model_id, model_id.title(), True) for model_id in MODEL_IDS]
        with patch("ui.ui_bridge.discover_models", return_value=statuses):
            self.bridge.start()
            self._wait_until(lambda: self.bridge.subtitle == online_caption(len(MODEL_IDS)))

    def tearDown(self):
        self.bridge.stop()
        events.reset_for_tests()

    def _pump(self):
        for _ in range(4):
            self.app.processEvents()

    def _wait_until(self, predicate, timeout=5.0):
        """Pump the event loop until `predicate` holds. The availability
        probe genuinely runs on a worker thread, so this waits for real
        work rather than assuming a fixed number of iterations."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return predicate()

    def _state(self, model_id):
        return {entry["id"]: entry["state"] for entry in self.bridge.models}[model_id]

    def test_the_subtitle_counts_real_availability(self):
        self.assertEqual(self.bridge.subtitle, online_caption(len(MODEL_IDS)))
        self.bridge.stop()  # no background re-probe may race this
        self.bridge.set_model_enabled("gemini", False, "not_implemented")
        self._pump()
        self.assertEqual(self.bridge.subtitle, online_caption(len(MODEL_IDS) - 1))

    def test_it_starts_without_claiming_zero_models(self):
        """The first probe runs on a worker thread and takes seconds in a
        cold process. "0 MODELS ONLINE" until it lands would be wrong, not
        merely empty."""
        fresh = UiBridge()
        self.assertNotIn("MODELS ONLINE", fresh.subtitle)

    def test_a_real_model_request_lights_the_node_it_belongs_to(self):
        events.publish(events.MODEL_REQUEST_STARTED, model="anthropic")
        self._pump()
        self.assertEqual(self._state("anthropic"), "thinking")
        self.assertTrue(self.bridge.processing)
        events.publish(events.MODEL_REQUEST_SUCCEEDED, model="anthropic")
        self._pump()
        self.assertEqual(self._state("anthropic"), "active")
        self.assertFalse(self.bridge.processing)

    def test_a_rate_limited_failure_is_amber_and_a_real_failure_is_red(self):
        events.publish(events.MODEL_REQUEST_FAILED, model="openai", error="RateLimitError")
        events.publish(events.MODEL_REQUEST_FAILED, model="vision", error="ProviderAuthError")
        self._pump()
        self.assertEqual(self._state("openai"), STATE_RATE_LIMITED)
        self.assertEqual(self._state("vision"), STATE_ERROR)

    def test_an_unavailable_module_never_lights_up_even_red(self):
        """The optional local intent service reports "not running" on
        every command when it is not installed. That is its normal state,
        not an alarm."""
        self.bridge.set_model_enabled("local", False, "service_not_running")
        self._pump()
        events.publish(events.MODEL_REQUEST_FAILED, model="local", error="service_unavailable")
        self._pump()
        self.assertEqual(self._state("local"), "offline")

    def test_assistant_state_drives_listening_speaking_and_processing(self):
        events.publish(events.ASSISTANT_STATE, state="LISTENING", detail="Listening")
        self._pump()
        self.assertTrue(self.bridge.listening)
        self.assertFalse(self.bridge.speaking)
        self.assertEqual(self.bridge.statusText, "Listening")

        events.publish(events.ASSISTANT_STATE, state="SPEAKING", detail="Speaking")
        self._pump()
        self.assertTrue(self.bridge.speaking)
        self.assertFalse(self.bridge.listening)

        events.publish(events.ASSISTANT_STATE, state="PROCESSING", detail="Working")
        self._pump()
        self.assertTrue(self.bridge.processing)

    def test_a_finished_model_call_cannot_clear_a_busy_assistant(self):
        events.publish(events.ASSISTANT_STATE, state="PROCESSING", detail="Working")
        events.publish(events.MODEL_REQUEST_STARTED, model="anthropic")
        self._pump()
        events.publish(events.MODEL_REQUEST_SUCCEEDED, model="anthropic")
        self._pump()
        self.assertTrue(self.bridge.processing)

    def test_transcripts_and_replies_reach_the_window(self):
        events.publish(events.USER_TEXT, text="open spotify")
        events.publish(events.JARVIS_TEXT, text="Opening Spotify, sir.")
        events.publish(events.STATUS_TEXT, text="Starting JARVIS")
        self._pump()
        self.assertEqual(self.bridge.userText, "open spotify")
        self.assertEqual(self.bridge.jarvisText, "Opening Spotify, sir.")
        self.assertEqual(self.bridge.statusText, "Starting JARVIS")

    def test_the_vendor_neutral_lifecycle_events_are_understood(self):
        """The provider/router work publishes its own neutral family. The
        window renders whichever of those name a node it draws."""
        self.assertIn(events.MODEL_RATE_LIMITED, NEUTRAL_EVENT_STATES)
        events.publish(events.MODEL_THINKING, provider="anthropic")
        self._pump()
        self.assertEqual(self._state("anthropic"), "thinking")
        events.publish(events.MODEL_RATE_LIMITED, model="openai")
        self._pump()
        self.assertEqual(self._state("openai"), STATE_RATE_LIMITED)

    def test_an_event_naming_an_unknown_capability_is_ignored_quietly(self):
        """A new capability or provider appearing on that bus must not
        make the window warn once per request."""
        with patch("ui.ui_bridge.log") as logger:
            events.publish(events.MODEL_THINKING, capability="reasoning")
            self._pump()
        logger.warning.assert_not_called()

    def test_a_subscriber_that_raises_cannot_break_the_publisher(self):
        """A display bug must never propagate into a live model call."""
        events.subscribe(None, Mock(side_effect=RuntimeError("ui exploded")))
        events.publish(events.MODEL_REQUEST_STARTED, model="anthropic")
        self._pump()
        self.assertEqual(self._state("anthropic"), "thinking")

    def test_an_unknown_model_id_or_state_is_ignored_rather_than_raised(self):
        self.bridge.set_model_state("not-a-model", "active")
        self.bridge.set_model_state("anthropic", "not-a-state")
        self._pump()
        self.assertEqual(self._state("anthropic"), "idle")

    def test_run_on_gui_thread_marshals_a_callable(self):
        seen = []
        self.bridge.run_on_gui_thread(lambda: seen.append(True))
        self._pump()
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
