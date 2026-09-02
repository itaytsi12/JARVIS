"""One JARVIS: voice and typed input share the vault. Milestone 16.

The requirement is that there is not a "voice JARVIS" and a "text JARVIS"
with separate memories. The proof is structural rather than rhetorical:
both surfaces already funnel through `brain/agent.py::run_agent`, so a
correction observed there is observed for both, and a mission started
below it is the same mission for both.

These tests drive the REAL `run_agent` with the real router, so they also
serve as the regression that the vault hooks did not disturb the existing
deterministic fast paths.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vault as vault_package
from vault.bootstrap import bootstrap_vault
from vault.index import VaultIndex
from vault.manager import VaultManager


class SharedSurfaceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        vault = VaultManager(self.root)
        bootstrap_vault(vault, VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0))
        vault_package.reset_all()
        self._patches = [
            patch("vault.paths.default_vault_path", return_value=self.root),
            patch("vault.paths.default_cache_path", return_value=Path(self.temp.name) / "cache.json"),
        ]
        for item in self._patches:
            item.start()
        vault_package.reset_all()

    def tearDown(self):
        for item in self._patches:
            item.stop()
        vault_package.reset_all()
        self.temp.cleanup()


class CorrectionOnTheSharedFunnelTests(SharedSurfaceTestCase):
    """`run_agent` is where BOTH surfaces meet, so it is where a
    correction has to be observed."""

    def _run(self, text, *, spoken=None):
        from brain.agent import run_agent

        # A deterministic route so nothing reaches a model. The point of
        # the test is the correction hook, which runs before routing.
        return run_agent(text, original_user_text=spoken)

    def test_a_typed_correction_updates_the_vault(self):
        self._run("From now on, always keep your spoken answers short.")
        preferences = VaultManager(self.root).read("user/preferences.md").section("Preferences")
        self.assertIn("short", preferences.lower())

    def test_a_spoken_correction_updates_the_same_vault(self):
        """`original_user_text` is set only by the voice path. The
        correction must land in exactly the same note."""
        self._run(
            "From now on, always use metric units.",
            spoken="From now on, always use metric units.",
        )
        preferences = VaultManager(self.root).read("user/preferences.md").section("Preferences")
        self.assertIn("metric", preferences.lower())

    def test_an_ordinary_command_writes_nothing(self):
        vault = VaultManager(self.root)
        before = vault.read("user/preferences.md").to_markdown()
        self._run("what time is it")
        self.assertEqual(vault.read("user/preferences.md").to_markdown(), before)

    def test_a_one_off_instruction_writes_nothing(self):
        vault = VaultManager(self.root)
        before = vault.read("user/preferences.md").to_markdown()
        self._run("Make this answer shorter.")
        self.assertEqual(vault.read("user/preferences.md").to_markdown(), before)

    def test_the_correction_is_recorded_in_todays_daily_note(self):
        self._run("From now on, always tell me the exit code.")
        from vault.daily import get_journal

        today = get_journal().today()
        corrections = VaultManager(self.root).read(today.relative_path).section(
            "User Corrections / Preferences Learned"
        )
        self.assertIn("exit code", corrections.lower())

    def test_a_vault_failure_never_breaks_a_command(self):
        with patch("vault.learning.get_correction_learner", side_effect=OSError("disk gone")):
            response = self._run("From now on, always use metric units.")
        self.assertIsInstance(response, str)


class FastPathRegressionTests(SharedSurfaceTestCase):
    """The deterministic routes still behave exactly as they did."""

    def test_a_simple_command_still_routes_deterministically(self):
        from brain.router import route_command

        route = route_command("volume down")
        self.assertEqual(route["type"], "tool")
        self.assertEqual(route["tool"], "volume_down")

    def test_a_music_command_still_routes_to_the_music_tools(self):
        from brain.router import route_command

        route = route_command("pause the music")
        self.assertEqual(route["type"], "tool")
        self.assertTrue(route["tool"].startswith("music_"), route["tool"])

    def test_routing_never_touches_the_vault(self):
        """Routing is the hot path. It must not pay for long-term memory."""
        from brain.router import route_command

        with patch("vault.manager.VaultManager.notes", side_effect=AssertionError("the router read the vault")):
            route_command("volume up")
            route_command("open notepad")


if __name__ == "__main__":
    unittest.main()
