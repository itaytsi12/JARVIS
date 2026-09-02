from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from voice.local_realtime_stt import LocalRealtimeSTTController
from voice.voice_perf import VoiceInteractionTimer


class LocalRealtimeSTTTests(unittest.TestCase):
    def test_never_opens_a_microphone_or_persists_partial_text(self):
        import voice.local_realtime_stt as module
        source = open(module.__file__, encoding="utf-8").read()
        self.assertNotIn("sounddevice", source)
        self.assertNotIn("RawInputStream", source)
        self.assertNotIn("training_data", source)

    def test_stable_safe_partial_uses_shared_speculative_ledger(self):
        fired = []
        controller = LocalRealtimeSTTController(
            perf=VoiceInteractionTimer(), on_speculative_action=fired.append,
            min_stable_partials=2, interval_seconds=0.01,
        )
        with patch.object(controller, "_transcribe", return_value="open notepad"):
            controller.start()
            controller.feed(b"\0\0" * 160)
            time.sleep(0.03)
            controller.feed(b"\0\0" * 160)
            deadline = time.time() + 1
            while not fired and time.time() < deadline:
                time.sleep(0.01)
            controller.close()
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].route["tool"], "open_application")
        self.assertTrue(controller.has_stable_partial)

    def test_destructive_partial_never_fires(self):
        fired = []
        controller = LocalRealtimeSTTController(
            perf=VoiceInteractionTimer(), on_speculative_action=fired.append,
            min_stable_partials=1,
        )
        controller._observe("send a message to John")
        self.assertEqual(fired, [])

    def test_final_full_snapshot_is_authoritative(self):
        controller = LocalRealtimeSTTController(perf=VoiceInteractionTimer())
        controller.feed(b"\0\0" * 160)
        with patch.object(controller, "_transcribe", return_value="final corrected text") as transcribe:
            result = controller.commit_and_close()
        self.assertEqual(result, "final corrected text")
        transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
