"""The standardized tool interface: schemas, validation, execution and failure.

No real desktop automation happens here -- execution is checked against
tools that are genuinely safe and deterministic (calculator, filesystem,
terminal) or against an injected handler.
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from brain.models import ActionRisk, ToolResult
from brain.tool_catalog import (
    CODE,
    DEFINITIONS,
    FILESYSTEM,
    TERMINAL,
    ToolCatalog,
    ToolDefinition,
    get_tool_catalog,
    validate_arguments,
)


class CatalogShapeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = get_tool_catalog()

    def test_every_definition_is_machine_readable(self):
        for definition in DEFINITIONS:
            with self.subTest(tool=definition.name):
                self.assertTrue(definition.description.strip())
                self.assertEqual(definition.parameters["type"], "object")
                self.assertIn("properties", definition.parameters)
                self.assertIn("required", definition.parameters)
                self.assertIsInstance(definition.risk, ActionRisk)

    def test_tool_names_are_unique(self):
        names = [definition.name for definition in DEFINITIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_required_arguments_are_declared_properties(self):
        for definition in DEFINITIONS:
            with self.subTest(tool=definition.name):
                declared = set(definition.parameters["properties"])
                self.assertTrue(set(definition.parameters["required"]) <= declared)

    def test_specs_convert_to_the_provider_format(self):
        spec = self.catalog.get("calculator").to_spec()
        self.assertEqual(spec.name, "calculator")
        self.assertIn("input_schema", spec.to_dict())

    def test_selection_by_category_returns_a_coherent_subset(self):
        names = {definition.name for definition in self.catalog.select(categories=[CODE, TERMINAL])}
        self.assertIn("run_command", names)
        self.assertIn("edit_code", names)
        self.assertNotIn("open_application", names)

    def test_destructive_and_ui_tools_are_classified(self):
        self.assertEqual(self.catalog.get("close_application").risk, ActionRisk.CAUTION)
        self.assertEqual(self.catalog.get("type_text").exclusive_resource, "desktop_input")
        self.assertTrue(self.catalog.get("read_text_file").read_only)

    def test_there_is_no_file_deletion_tool(self):
        # Deletion is deliberately absent: reading is safe, destroying is not.
        names = {definition.name for definition in DEFINITIONS}
        self.assertNotIn("delete_file", names)
        self.assertNotIn("delete_path", names)


class ArgumentValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = get_tool_catalog()

    def test_missing_required_argument_is_reported_by_name(self):
        errors = validate_arguments(self.catalog.get("calculator"), {})
        self.assertIn("missing_required:expression", errors)

    def test_unknown_argument_is_rejected(self):
        errors = validate_arguments(self.catalog.get("calculator"), {"expression": "1+1", "extra": 1})
        self.assertIn("unknown_argument:extra", errors)

    def test_wrong_type_is_rejected(self):
        errors = validate_arguments(self.catalog.get("read_code"), {"path": "x.py", "start_line": "one"})
        self.assertIn("invalid_type:start_line", errors)

    def test_boolean_is_not_accepted_as_an_integer(self):
        errors = validate_arguments(self.catalog.get("read_code"), {"path": "x.py", "start_line": True})
        self.assertIn("invalid_type:start_line", errors)

    def test_valid_arguments_produce_no_errors(self):
        self.assertEqual(validate_arguments(self.catalog.get("read_code"), {"path": "x.py", "start_line": 3}), [])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog()
        self.root = Path(tempfile.mkdtemp())

    def test_a_successful_tool_returns_a_tool_result(self):
        result = self.catalog.execute("calculator", {"expression": "527*93"})
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.message, "49011")
        self.assertIn("duration_ms", result.data)
        self.assertEqual(result.data["category"], "info")

    def test_an_unknown_tool_fails_without_raising(self):
        result = self.catalog.execute("teleport", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "unknown_tool")
        self.assertIn("available_tools", result.data)

    def test_invalid_arguments_fail_with_the_schema_attached(self):
        result = self.catalog.execute("calculator", {"expr": "1+1"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_arguments")
        self.assertIn("schema", result.data)

    def test_a_tool_that_fails_reports_its_own_error(self):
        result = self.catalog.execute("read_text_file", {"path": str(self.root / "missing.txt")})
        self.assertFalse(result.success)
        self.assertTrue(result.error)

    def test_a_raising_tool_is_isolated_into_a_failed_result(self):
        def explode(arguments):
            raise RuntimeError("kaboom")

        self.catalog.register_handler("calculator", explode)
        result = self.catalog.execute("calculator", {"expression": "1+1"})
        self.assertFalse(result.success)
        self.assertIn("kaboom", result.error)

    def test_a_cancelled_token_stops_execution_before_the_tool_runs(self):
        class Token:
            cancelled = True

        marker = self.root / "should-not-exist.txt"
        result = self.catalog.execute(
            "create_text_file", {"path": str(marker), "contents": "x"}, cancellation_token=Token()
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "cancelled")
        self.assertFalse(marker.exists())

    def test_registered_handler_takes_precedence(self):
        self.catalog.register_handler(
            "remember_fact", lambda arguments: ToolResult(True, "remember_fact", f"stored {arguments['text']}")
        )
        result = self.catalog.execute("remember_fact", {"text": "hello"})
        self.assertTrue(result.success)
        self.assertEqual(result.message, "stored hello")

    def test_filesystem_round_trip_through_the_catalog(self):
        target = self.root / "notes" / "a.txt"
        self.assertTrue(self.catalog.execute("create_directory", {"path": str(target.parent)}).success)
        self.assertTrue(self.catalog.execute("create_text_file", {"path": str(target), "contents": "hello"}).success)
        read = self.catalog.execute("read_text_file", {"path": str(target)})
        self.assertTrue(read.success)
        self.assertEqual(read.data["contents"], "hello")

    def test_custom_definitions_are_honoured(self):
        catalog = ToolCatalog([ToolDefinition("only", "only tool", {"type": "object", "properties": {}, "required": []}, FILESYSTEM)])
        self.assertEqual(catalog.names(), ["only"])


class ConcurrencyIsolationTests(unittest.TestCase):
    """Independent work must not serialize behind the desktop lock, and
    desktop work must still serialize."""

    def setUp(self):
        self.catalog = ToolCatalog()
        self.work = Path(tempfile.mkdtemp())
        (self.work / "sleep.py").write_text("import time\ntime.sleep(0.4)\n", encoding="utf-8")
        self.command = f'"{sys.executable}" sleep.py'
        # The first tool call imports the executor chain; warm it up so the
        # measurement below reflects locking, not one-off import cost.
        self.catalog.execute("run_command", {"command": self.command, "working_directory": str(self.work)})

    def _run_concurrently(self, count: int) -> float:
        def work():
            self.catalog.execute("run_command", {"command": self.command, "working_directory": str(self.work)})

        threads = [threading.Thread(target=work) for _ in range(count)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.perf_counter() - started

    def test_terminal_work_is_not_serialized_by_the_desktop_lock(self):
        elapsed = self._run_concurrently(3)
        self.assertLess(elapsed, 1.1, f"3 x 0.4s commands took {elapsed:.2f}s -- they are serializing")

    def test_desktop_tools_still_take_the_process_wide_plan_lock(self):
        from brain.tool_catalog import SESSION_AWARE_CATEGORIES, SESSION_AWARE_TOOLS

        catalog = get_tool_catalog()
        for name in ("type_text", "press_key", "click_ui_element", "open_application", "browser_open_url", "open_path"):
            with self.subTest(tool=name):
                definition = catalog.get(name)
                self.assertTrue(
                    definition.category in SESSION_AWARE_CATEGORIES or definition.name in SESSION_AWARE_TOOLS,
                    f"{name} must stay on the session-aware, fully-locked path",
                )

    def test_read_only_work_is_not_session_aware(self):
        from brain.tool_catalog import SESSION_AWARE_CATEGORIES, SESSION_AWARE_TOOLS

        catalog = get_tool_catalog()
        for name in ("read_code", "search_code", "run_command", "read_text_file", "recall_memory"):
            with self.subTest(tool=name):
                definition = catalog.get(name)
                self.assertNotIn(definition.category, SESSION_AWARE_CATEGORIES)
                self.assertNotIn(definition.name, SESSION_AWARE_TOOLS)


if __name__ == "__main__":
    unittest.main()
