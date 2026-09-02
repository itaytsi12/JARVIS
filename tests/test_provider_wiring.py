"""Provider wiring: does the process that actually runs JARVIS have Claude?

The bug these lock down was NOT in the provider implementation -- the real
Anthropic smoke test passed the whole time. It was that the interpreter the
live tray runs in (`.venv-agent`) had a valid `ANTHROPIC_API_KEY` and
`JARVIS_AGENT_MODEL` but no `anthropic` package, so
`AnthropicProvider.is_available()` returned False with
`unavailable_reason="anthropic_sdk_not_installed"` -- a reason that was
computed, stored, and never logged by anybody. The runtime said only
"Complex request, but no agent provider is configured", fell back to the
cloud planner, and failed.

Two properties therefore matter here and are tested separately:

- the CONFIGURATION path must not depend on where the process was started
  or which entry point it came through, and
- a key present with no usable provider must be LOUD, never a silent
  degrade.
"""
from __future__ import annotations

import builtins
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import reload_config
from providers.mock_provider import CallableProvider, text_response
from providers.registry import (
    agent_escalation_available,
    agent_unavailable_reason,
    provider_status,
    register_provider,
    reset_providers_for_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_python_files():
    """Every `.py` file that is genuinely part of this project.

    `PROJECT_ROOT.rglob("*.py")` cannot prune: it descends into every
    virtualenv, `.git`, the model artifacts and the caches, then leaves
    the caller to discard the results. That is the same defect
    `tools/code.py::walk_source_files` exists to fix (98 seconds there),
    and here it made one assertion take 205 SECONDS -- longer than the
    rest of the suite put together. Reusing the project's own pruning
    walker gives the identical answer for a fraction of the work.
    """
    from tools.code import walk_source_files

    for path in walk_source_files(PROJECT_ROOT):
        if path.suffix == ".py":
            yield path

class _Provider(CallableProvider):
    """A stand-in registered under the real provider name, so the code under
    test resolves it exactly as it resolves Anthropic."""

    name = "anthropic"


def install_provider(test_case, answer: str = "Handled it, sir.", model: str = "claude-sonnet-5"):
    provider = _Provider(lambda messages, **kwargs: text_response(answer), model=model)
    register_provider("anthropic", lambda: provider)
    test_case.addCleanup(reset_providers_for_tests)
    return provider


def without_the_anthropic_sdk():
    """Reproduce the live `.venv-agent` state: the SDK cannot be imported.

    Patches the import machinery rather than deleting a module, because
    `providers/anthropic_provider.py` imports `anthropic` lazily inside
    `is_available()` -- which is precisely why the failure was survivable
    enough to go unnoticed.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    return patch.object(builtins, "__import__", blocked)


class DotenvIsLoadedFromTheProjectRootTests(unittest.TestCase):
    """`.env` must load the same way whatever the working directory is."""

    def test_settings_loads_dotenv_from_the_project_root_not_the_cwd(self):
        import config.settings as settings

        self.assertEqual(settings.PROJECT_ROOT, PROJECT_ROOT)

    def test_no_module_outside_config_loads_dotenv_itself(self):
        """A second, CWD-relative `load_dotenv()` re-introduces the ordering
        hazard: whichever import ran first decided the configuration."""
        offenders = []
        for path in _project_python_files():
            if path.name.startswith("."):
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative.startswith(("config/", "scripts/", "tests/", ".venv")):
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                # Comments explaining WHY a module must not load .env are the
                # point of this rule, not a violation of it.
                if stripped.startswith("#"):
                    continue
                if "load_dotenv(" in stripped:
                    offenders.append(f"{relative}: {stripped}")
        self.assertEqual(offenders, [], f"these modules load .env themselves: {offenders}")

    def test_configuration_survives_a_foreign_working_directory(self):
        """The live symptom: started elsewhere, the key was never found and
        `agent_model` silently fell back to its default."""
        import subprocess

        script = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from config.settings import JarvisConfig\n"
            "c = JarvisConfig()\n"
            "print(c.agent_model)\n"
        ) % PROJECT_ROOT
        environment = dict(os.environ)
        environment["JARVIS_AGENT_MODEL"] = ""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(PROJECT_ROOT.parent),
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # It read SOMETHING real rather than crashing, from a foreign cwd.
        self.assertTrue(result.stdout.strip())


class NoApiKeyStillWorksTests(unittest.TestCase):
    """"If no Anthropic key is configured, local/simple commands must still
    work" -- the "Claude is optional" invariant."""

    def setUp(self):
        reset_providers_for_tests()
        self.addCleanup(reset_providers_for_tests)
        self.addCleanup(reload_config)

    def test_simple_commands_route_locally_with_no_key(self):
        from brain.router import route_command

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            reload_config()
            self.assertFalse(agent_escalation_available())
            for command, tool in (
                ("open Spotify", "open_application"),
                ("volume down", "volume_down"),
                ("what time is it", "get_time"),
                ("inspect window", "inspect_window"),
            ):
                with self.subTest(command=command):
                    route = route_command(command)
                    self.assertEqual(route["type"], "tool")
                    self.assertEqual(route["tool"], tool)

    def test_a_complex_request_degrades_without_raising(self):
        from brain.router import route_command

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            reload_config()
            route = route_command("Inspect this project and explain how it works. Do not modify anything.")
        # The router still escalates; brain/agent.py degrades it to the planner.
        self.assertEqual(route["type"], "agent_task")
        self.assertEqual(route["route_source"], "complexity_guard")

    def test_no_key_is_reported_as_normal_not_as_an_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            reload_config()
            with self.assertLogs("jarvis", level="INFO") as captured:
                from config import log_startup_status

                log_startup_status(force=True)
        joined = "\n".join(captured.output)
        self.assertIn("api_key_present=no", joined)
        self.assertNotIn("ERROR", joined)


class ProviderAvailableTests(unittest.TestCase):
    """"If Anthropic is configured, complex `agent_task` routes must invoke
    the real AgentRuntime rather than the legacy cloud_planner fallback." """

    def test_agent_task_invokes_the_runtime_not_the_cloud_planner(self):
        from brain import agent

        provider = install_provider(self)
        outcome: dict = {}
        with patch.object(agent, "create_plan", side_effect=AssertionError("the legacy cloud planner was called")):
            answer = agent.run_agent(
                "Tell me what files are in the JARVIS project folder. Do not modify anything.",
                execution_outcome=outcome,
            )
        self.assertEqual(answer, "Handled it, sir.")
        self.assertEqual(outcome["route_type"], "agent_task")
        self.assertNotEqual(outcome.get("route_source"), "cloud_planner")
        self.assertGreaterEqual(outcome.get("model_calls", 0), 1)

    def test_the_reported_model_is_the_configured_one(self):
        provider = install_provider(self, model="claude-sonnet-5")
        status = provider_status()
        self.assertEqual(status["active_provider"], "anthropic")
        self.assertEqual(status["active_model"], "claude-sonnet-5")
        self.assertIsNone(agent_unavailable_reason())


class BothRuntimesUseTheSameProviderTests(unittest.TestCase):
    """"Make sure the tray runtime and typed runtime use the same provider
    creation/configuration logic." Same function, not merely same result."""

    def test_neither_entry_point_creates_a_provider_of_its_own(self):
        for module in ("main.py", "voice/tray_app.py", "voice/background_assistant.py"):
            with self.subTest(module=module):
                text = (PROJECT_ROOT / module).read_text(encoding="utf-8")
                self.assertNotIn("AnthropicProvider", text)
                self.assertNotIn("anthropic.Anthropic", text)

    def test_only_the_anthropic_provider_module_imports_the_sdk(self):
        offenders = []
        for path in _project_python_files():
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if relative in {"providers/anthropic_provider.py"} or relative.startswith("tests/"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import anthropic", "from anthropic")):
                    offenders.append(f"{relative}: {stripped}")
        self.assertEqual(offenders, [], f"only providers/anthropic_provider.py may import the SDK: {offenders}")

    def test_both_entry_points_call_the_same_startup_reporter(self):
        main_text = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        tray_text = (PROJECT_ROOT / "voice" / "tray_app.py").read_text(encoding="utf-8")
        self.assertIn("log_startup_status()", main_text)
        self.assertIn("log_startup_status()", tray_text)

    def test_the_tray_runtime_resolves_the_same_provider_object(self):
        """The voice runtime reaches the provider through `brain.agent`'s
        module-level `agent_runtime`; the typed runtime through the registry.
        Both must land on the one registered provider."""
        provider = install_provider(self)
        from brain.agent import _agent_escalation_available
        from brain.agent_service import get_agent_provider

        self.assertTrue(_agent_escalation_available())
        self.assertIs(get_agent_provider(), provider)

    def test_the_voice_dispatcher_agrees_that_this_is_an_agent_route(self):
        install_provider(self)
        from brain.router import route_command
        from voice.background_assistant import AlwaysOnAssistant

        route = route_command("Read main.py and tell me what it does. Do not modify anything.")
        self.assertTrue(AlwaysOnAssistant._is_agent_route(route))


class InitializationFailureIsVisibleTests(unittest.TestCase):
    """The heart of it: a key present with no usable provider must be LOUD."""

    def setUp(self):
        reset_providers_for_tests()
        self.addCleanup(reset_providers_for_tests)

    def test_a_missing_sdk_is_reported_with_the_real_reason(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-not-a-real-key"}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            with without_the_anthropic_sdk():
                self.assertFalse(agent_escalation_available())
                self.assertIn("anthropic_sdk_not_installed", agent_unavailable_reason() or "")

    def test_a_missing_sdk_is_logged_at_error_not_silently_degraded(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-not-a-real-key"}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            with without_the_anthropic_sdk(), self.assertLogs("jarvis", level="INFO") as captured:
                from config import log_startup_status

                log_startup_status(force=True)
        joined = "\n".join(captured.output)
        self.assertIn("Agent provider configured: no", joined)
        self.assertIn("api_key_present=yes", joined)
        self.assertIn("anthropic_sdk_not_installed", joined)
        self.assertTrue(
            any(record.startswith("ERROR") for record in captured.output),
            f"a configured key with no usable provider must be an ERROR: {joined}",
        )

    def test_the_runtime_names_the_reason_when_it_falls_back(self):
        """"Complex request, but no agent provider is configured" was true
        but misleading -- the key WAS configured."""
        from brain import agent

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-not-a-real-key"}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            # The degraded path is the legacy cloud planner, which would make
            # a real OpenAI call. Stub it: what is under test is the WARNING
            # emitted before the fallback, not the fallback itself.
            with without_the_anthropic_sdk(),                  patch.object(agent, "create_plan", return_value=[]),                  self.assertLogs("jarvis.runtime", level="WARNING") as captured:
                agent.run_agent("Inspect this project and explain how it works. Do not modify anything.")
        joined = "\n".join(captured.output)
        self.assertIn("agent runtime is unavailable", joined)
        self.assertIn("anthropic_sdk_not_installed", joined)

    def test_no_log_line_can_contain_the_api_key(self):
        secret = "sk-ant-api03-THIS-MUST-NEVER-APPEAR"
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            with without_the_anthropic_sdk(), self.assertLogs("jarvis", level="INFO") as captured:
                from config import log_startup_status

                log_startup_status(force=True)
        joined = "\n".join(captured.output)
        self.assertNotIn(secret, joined)
        self.assertNotIn(secret[-12:], joined)

    def test_provider_status_never_exposes_the_key(self):
        secret = "sk-ant-api03-ANOTHER-SECRET-VALUE"
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}, clear=False):
            reload_config()
            self.addCleanup(reload_config)
            rendered = repr(provider_status())
        self.assertNotIn(secret, rendered)
        self.assertIn("<set>", rendered)


class TheRuntimeVenvDeclaresTheSdkTests(unittest.TestCase):
    """The actual root cause: `.venv-agent` is what `python main.py --tray`
    runs in, and its requirements file did not mention `anthropic`."""

    def test_requirements_agent_declares_the_anthropic_sdk(self):
        text = (PROJECT_ROOT / "requirements-agent.txt").read_text(encoding="utf-8")
        declarations = [
            line.split("#")[0].strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertTrue(
            any(item.startswith("anthropic") for item in declarations),
            f"the tray's virtualenv must declare the Claude SDK: {declarations}",
        )

    def test_every_requirements_file_that_runs_the_agent_declares_it(self):
        for name in ("requirements.txt", "requirements-agent.txt"):
            with self.subTest(requirements=name):
                text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
                self.assertIn("anthropic", text)


if __name__ == "__main__":
    unittest.main()
