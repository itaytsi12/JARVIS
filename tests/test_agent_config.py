"""Configuration, pricing and the no-API-key path."""
import os
import unittest
from unittest.mock import patch

from config.pricing import estimate_cost, pricing_for, reload_pricing_table
from config.settings import JarvisConfig, reload_config


class ConfigurationTests(unittest.TestCase):
    def tearDown(self):
        reload_config()

    def test_defaults_work_with_an_empty_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            config = JarvisConfig()
        self.assertTrue(config.agent_model)
        self.assertGreater(config.max_agent_steps, 0)
        self.assertGreater(config.max_concurrent_tasks, 0)
        self.assertFalse(config.has_anthropic_credentials)

    def test_environment_overrides_are_applied(self):
        with patch.dict(os.environ, {"JARVIS_MAX_AGENT_STEPS": "7", "JARVIS_AGENT_MODEL": "claude-sonnet-5"}):
            config = reload_config()
        self.assertEqual(config.max_agent_steps, 7)
        self.assertEqual(config.agent_model, "claude-sonnet-5")

    def test_malformed_numeric_setting_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"JARVIS_MAX_AGENT_STEPS": "not-a-number"}):
            config = reload_config()
        self.assertEqual(config.max_agent_steps, 25)

    def test_describe_never_exposes_the_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-super-secret-value"}):
            config = reload_config()
            described = config.describe()
        self.assertEqual(described["anthropic_api_key"], "<set>")
        self.assertNotIn("sk-ant-super-secret-value", repr(described))
        self.assertNotIn("sk-ant-super-secret-value", repr(config))

    def test_missing_key_is_reported_as_unset_not_as_an_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            config = reload_config()
        self.assertFalse(config.has_anthropic_credentials)
        self.assertEqual(config.describe()["anthropic_api_key"], "<unset>")

    def test_blank_key_is_treated_as_absent(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "   "}):
            config = reload_config()
        self.assertIsNone(config.anthropic_api_key)


class PricingTests(unittest.TestCase):
    def test_known_model_is_priced(self):
        self.assertAlmostEqual(estimate_cost("claude-opus-5", 1_000_000, 1_000_000), 30.0, places=6)

    def test_unknown_model_returns_none_rather_than_zero(self):
        # "unknown" and "free" must stay distinguishable.
        self.assertIsNone(estimate_cost("some-local-model", 1000, 1000))

    def test_longest_prefix_wins(self):
        self.assertEqual(pricing_for("claude-opus-4-8-something"), pricing_for("claude-opus-4-8"))

    def test_cache_tokens_are_priced_when_reported(self):
        without = estimate_cost("claude-sonnet-5", 1000, 0)
        with_cache = estimate_cost("claude-sonnet-5", 1000, 0, cache_read_tokens=1_000_000)
        self.assertGreater(with_cache, without)

    def test_pricing_file_override(self):
        import json
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "pricing.json"
        path.write_text(json.dumps({"my-local-model": {"input_per_mtok": 0.5, "output_per_mtok": 1.0}}), encoding="utf-8")
        with patch.dict(os.environ, {"JARVIS_PRICING_FILE": str(path)}):
            reload_pricing_table()
            self.assertAlmostEqual(estimate_cost("my-local-model", 1_000_000, 0), 0.5, places=6)
        reload_pricing_table()


if __name__ == "__main__":
    unittest.main()
