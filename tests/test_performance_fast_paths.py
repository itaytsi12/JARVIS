import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brain import agent
from brain import router
from brain.router import route_command
from brain import local_intent_model
from brain import intent_router
from urllib.error import URLError


class PerformanceFastPathTests(unittest.TestCase):
    def test_common_deterministic_routes_stay_local_and_fast(self):
        commands = [
            "Open YouTube",
            "Open Notepad",
            "Turn the volume down",
            "Search YouTube for Jude Law",
        ]
        started = time.perf_counter()
        routes = [route_command(command) for command in commands]
        elapsed = time.perf_counter() - started

        self.assertTrue(all(route["type"] == "tool" for route in routes))
        # Generous guardrail: this is intended to catch accidental model/API paths.
        self.assertLess(elapsed, 0.5)

    def test_precomputed_route_is_not_classified_again(self):
        route = {"type": "tool", "tool": "open_application", "arguments": {"app_name": "notepad"}}
        result = Mock(success=True, message="Opened notepad successfully.")
        with patch.object(agent, "route_command") as classify, patch.object(agent.executor, "execute_action", return_value=result):
            self.assertEqual(agent.run_agent("Open Notepad", route=route), result.message)
        classify.assert_not_called()

    def test_voice_browser_phrasing_routes_to_websites(self):
        cases = {
            "Open YouTube on Google.": "https://www.youtube.com",
            "Open TikTok on Google.": "https://www.tiktok.com",
            "Open TikTok": "https://www.tiktok.com",
        }
        for command, expected_url in cases.items():
            with self.subTest(command=command):
                route = route_command(command)
                self.assertEqual(route["tool"], "open_website")
                self.assertEqual(route["arguments"]["url"], expected_url)

    def test_unavailable_local_intent_service_has_bounded_fallback_timeout(self):
        with patch.object(local_intent_model,"urlopen",side_effect=URLError("offline")) as request:
            self.assertIsNone(local_intent_model.predict_local_intent("an unknown request"))
        self.assertEqual(request.call_args.kwargs["timeout"],local_intent_model.INTENT_SERVICE_TIMEOUT)
        self.assertLessEqual(local_intent_model.INTENT_SERVICE_TIMEOUT,.5)

    def test_cloud_router_fallback_carries_model_provenance(self):
        with patch.object(router,"route_with_local_model",return_value=None),patch.object(router,"classify_intent",return_value={"type":"ai","message":"frobnicate softly"}):
            route=router.route_command("frobnicate softly")
        self.assertEqual(route["route_source"],"cloud_intent_router");self.assertEqual(route["model_calls"],1)
        self.assertEqual(route["fallback_from"],["local_learned_classifier"]);self.assertEqual(route["fallback_reason"],"no_confident_local_route")

    def test_cloud_intent_router_preserves_reported_token_usage(self):
        response=SimpleNamespace(output=[],usage=SimpleNamespace(input_tokens=41,output_tokens=3))
        with patch.object(intent_router.client.responses,"create",return_value=response):route=intent_router.classify_intent("unknown request")
        self.assertEqual(route["model_calls"],1);self.assertEqual((route["input_tokens"],route["output_tokens"]),(41,3))

if __name__ == "__main__":
    unittest.main()
