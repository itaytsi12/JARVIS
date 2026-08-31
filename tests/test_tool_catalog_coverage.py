"""The catalog is the agent's whole view of what JARVIS can do.

A tool that `brain/tool_router.py` can dispatch but `brain/tool_catalog.py`
does not describe is invisible: the model is never told it exists, so it
can never be used no matter how well it works. That is not hypothetical --
the entire Apple Music family (18 tools, fully implemented and tested) was
in exactly that state, which is why "open Apple Music and make me a
playlist" could not work at all. These tests are the invariant that stops
it happening again in either direction.
"""
import re
import unittest
from pathlib import Path

from brain.models import ActionRisk
from brain.tool_catalog import BY_NAME, DEFINITIONS, ToolCatalog, get_tool_catalog, validate_arguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Dispatchable on purpose, and deliberately NOT offered to the agent.
#: Locking the machine is something the user asks for explicitly, and
#: sending a message is irreversible and outward-facing -- both keep their
#: own routes. Listing them here makes the exclusion a decision rather
#: than an oversight, and makes adding a third one a deliberate act.
INTENTIONALLY_UNDESCRIBED = {"lock_computer", "send_whatsapp_message"}

#: Dict keys in the router's result payloads, not tool names.
_RESULT_KEYS = {"error", "message", "success", "verified", "hwnd"}


def _dispatchable_tool_names() -> set[str]:
    source = (PROJECT_ROOT / "brain" / "tool_router.py").read_text(encoding="utf-8")
    names = set(re.findall(r'tool_name == "([a-z_0-9]+)"', source))
    names |= set(re.findall(r'^\s+"([a-z_0-9]+)":\s', source, re.MULTILINE))
    return names - _RESULT_KEYS


class CatalogCoverageTests(unittest.TestCase):
    def test_every_dispatchable_tool_is_described_to_the_agent(self):
        missing = _dispatchable_tool_names() - set(BY_NAME) - INTENTIONALLY_UNDESCRIBED
        self.assertEqual(
            missing,
            set(),
            "These tools can be dispatched but the agent is never told they exist, "
            "so it can never call them: " + ", ".join(sorted(missing)),
        )

    def test_the_music_tools_are_reachable_by_the_agent(self):
        """The specific regression: a full, working capability that the
        model had no way to discover."""
        for name in (
            "open_music", "music_play", "music_pause", "music_resume", "music_next",
            "music_previous", "music_now_playing", "music_list_playlists",
            "music_create_playlist", "music_add_to_playlist",
        ):
            self.assertIn(name, BY_NAME, f"{name} is not offered to the agent")

    def test_no_tool_is_declared_twice(self):
        self.assertEqual(len(DEFINITIONS), len(BY_NAME))

    def test_every_definition_has_a_usable_schema(self):
        for definition in DEFINITIONS:
            schema = definition.parameters
            self.assertEqual(schema.get("type"), "object", definition.name)
            self.assertIn("properties", schema, definition.name)
            self.assertFalse(schema.get("additionalProperties", False), definition.name)
            for required in schema.get("required", []):
                self.assertIn(required, schema["properties"], f"{definition.name}.{required}")

    def test_every_description_says_something_useful(self):
        for definition in DEFINITIONS:
            self.assertGreaterEqual(len(definition.description), 20, definition.name)
            self.assertTrue(definition.description.strip().endswith("."), definition.name)


class ToolMetadataTests(unittest.TestCase):
    def test_a_read_only_tool_is_always_retry_safe(self):
        """Reading twice is reading once. Normalised in `__post_init__` so
        no call site has to remember it."""
        for definition in DEFINITIONS:
            if definition.read_only:
                self.assertTrue(definition.retry_safe, definition.name)

    def test_a_read_only_tool_never_claims_an_exclusive_desktop_resource(self):
        for definition in DEFINITIONS:
            if definition.read_only:
                self.assertNotEqual(definition.exclusive_resource, "desktop_input", definition.name)

    def test_appending_and_relative_volume_are_not_retry_safe(self):
        """A retry duplicates the text, or moves the volume twice. These
        are the cases the flag exists to distinguish."""
        for name in ("append_text_file", "volume_up", "volume_down"):
            self.assertFalse(BY_NAME[name].retry_safe, name)

    def test_idempotent_writers_are_retry_safe(self):
        for name in ("set_volume", "write_clipboard", "scroll_screen"):
            self.assertTrue(BY_NAME[name].retry_safe, name)

    def test_every_timeout_is_a_positive_and_plausible_number(self):
        for definition in DEFINITIONS:
            self.assertGreater(definition.timeout_seconds, 0, definition.name)
            self.assertLessEqual(definition.timeout_seconds, 300, definition.name)

    def test_a_risky_tool_is_never_marked_read_only(self):
        for definition in DEFINITIONS:
            if definition.risk is not ActionRisk.SAFE:
                self.assertFalse(definition.read_only, definition.name)

    def test_describe_exposes_the_metadata_the_agent_needs(self):
        described = BY_NAME["recent_files"].describe()
        for key in ("name", "description", "category", "risk", "read_only", "retry_safe", "timeout_seconds", "parameters"):
            self.assertIn(key, described)


class ArgumentValidationTests(unittest.TestCase):
    def test_a_missing_required_argument_is_named_precisely(self):
        errors = validate_arguments(BY_NAME["music_add_to_playlist"], {"song": "Ordinary"})
        self.assertIn("missing_required:playlist", errors)

    def test_an_invented_argument_is_rejected_rather_than_ignored(self):
        errors = validate_arguments(BY_NAME["system_status"], {"detailed": True})
        self.assertIn("unknown_argument:detailed", errors)

    def test_a_boolean_is_not_accepted_where_an_integer_is_required(self):
        errors = validate_arguments(BY_NAME["set_volume"], {"level": True})
        self.assertIn("invalid_type:level", errors)

    def test_a_valid_call_produces_no_errors(self):
        self.assertEqual(validate_arguments(BY_NAME["set_volume"], {"level": 30}), [])
        self.assertEqual(validate_arguments(BY_NAME["recent_files"], {"within_hours": 12.5}), [])


class CatalogExecutionTests(unittest.TestCase):
    """Execution stays inside the catalog's contract: it never raises, and
    it always returns a `ToolResult` the agent loop can read."""

    def test_an_unknown_tool_returns_a_result_not_an_exception(self):
        result = ToolCatalog().execute("no_such_tool", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "unknown_tool")
        self.assertTrue(result.data["available_tools"])

    def test_invalid_arguments_come_back_with_the_schema_to_fix_them(self):
        result = get_tool_catalog().execute("set_volume", {"level": "loud"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_arguments")
        self.assertIn("schema", result.data)

    def test_a_tool_that_raises_becomes_a_failed_result(self):
        catalog = ToolCatalog()
        catalog.register_handler("system_status", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
        result = catalog.execute("system_status", {})
        self.assertFalse(result.success)
        self.assertIn("RuntimeError", result.error)

    def test_every_result_carries_the_metadata_the_loop_records(self):
        catalog = ToolCatalog()
        catalog.register_handler("get_date", lambda args: __import__("brain.models", fromlist=["ToolResult"]).ToolResult(True, "get_date", "ok"))
        result = catalog.execute("get_date", {})
        self.assertIn("duration_ms", result.data)
        self.assertEqual(result.data["category"], "info")
        self.assertIn("verified", result.data)


if __name__ == "__main__":
    unittest.main()
