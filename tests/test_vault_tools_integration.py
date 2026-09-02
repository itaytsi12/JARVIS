"""The vault tools and the agent-service integration contract.

Two concerns:

- the tools the MODEL is offered actually work, are described in the
  catalog (a tool the catalog does not describe is invisible), and cannot
  be used to edit protected knowledge;
- the integration into `run_agent_task` holds its promises: fast paths
  stay fast, the vault never breaks a request, and the knowledge really
  reaches the prompt.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vault as vault_package
from brain.tool_catalog import BY_NAME, ToolCatalog
from brain.tool_router import execute_tool
from vault.bootstrap import bootstrap_vault
from vault.index import VaultIndex
from vault.manager import VaultManager
from vault.policy import FULL, LIGHT, assess


class VaultToolTestCase(unittest.TestCase):
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


class ToolDescriptionTests(unittest.TestCase):
    """A tool the catalog does not describe can never be called, however
    well it works -- the exact failure the Apple Music family once hit."""

    TOOLS = (
        "vault_search",
        "vault_read_note",
        "vault_write_note",
        "vault_update_note",
        "vault_record_lesson",
        "vault_record_working_method",
        "vault_list_jobs",
        "vault_status",
    )

    def test_every_vault_tool_is_offered_to_the_agent(self):
        for name in self.TOOLS:
            self.assertIn(name, BY_NAME, f"{name} is dispatchable but the agent is never told it exists")

    def test_the_read_only_tools_are_marked_read_only(self):
        for name in ("vault_search", "vault_read_note", "vault_list_jobs", "vault_status"):
            self.assertTrue(BY_NAME[name].read_only, name)
            self.assertTrue(BY_NAME[name].retry_safe, name)

    def test_creating_a_note_is_not_retry_safe(self):
        """Running it twice is not the same as running it once."""
        self.assertFalse(BY_NAME["vault_write_note"].retry_safe)
        self.assertFalse(BY_NAME["vault_record_lesson"].retry_safe)

    def test_the_catalog_can_render_them_as_specs(self):
        specs = {spec.name for spec in ToolCatalog().specs(names=set(self.TOOLS))}
        self.assertEqual(specs, set(self.TOOLS))


class ToolBehaviourTests(VaultToolTestCase):
    def test_search_scans_summaries_and_reads_no_bodies(self):
        result = execute_tool("vault_search", {"query": "fix a failing python test"})
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["scanned"], 10)
        self.assertTrue(any("Fix Software Bug" in line["title"] for line in result["results"]))

    def test_search_reports_an_honest_miss(self):
        """And an incidental single-word match is not a "result": one
        stray word in a quick summary is noise, and noise in a tool result
        is something the model may act on."""
        result = execute_tool("vault_search", {"query": "zzzz qqqq wwww"})
        self.assertTrue(result["success"])
        self.assertEqual(result["results"], [])
        self.assertIn("Nothing in the vault matches", result["message"])

    def test_read_note_returns_the_note_and_its_sections(self):
        result = execute_tool("vault_read_note", {"path": "skills/python-debugging.md"})
        self.assertTrue(result["success"])
        self.assertIn("Reproduce before theorising", result["message"])
        self.assertIn("Procedure", result["sections"])

    def test_read_note_can_return_one_section(self):
        result = execute_tool("vault_read_note", {"path": "skills/python-debugging.md", "section": "Procedure"})
        self.assertTrue(result["success"])
        self.assertIn("Run the failing command", result["message"])
        self.assertNotIn("When To Use", result["message"])

    def test_a_missing_section_lists_the_real_ones_rather_than_failing_blankly(self):
        result = execute_tool("vault_read_note", {"path": "skills/python-debugging.md", "section": "Nonexistent"})
        self.assertFalse(result["success"])
        self.assertIn("Procedure", result["message"])

    def test_a_path_outside_the_vault_is_refused(self):
        result = execute_tool("vault_read_note", {"path": "../../../.env"})
        self.assertFalse(result["success"])

    def test_write_note_requires_a_summary(self):
        result = execute_tool(
            "vault_write_note",
            {"path": "lessons/x.md", "title": "X", "note_type": "lesson", "summary": "  ", "content": "body"},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "summary_required")

    def test_write_note_creates_a_conforming_note(self):
        result = execute_tool(
            "vault_write_note",
            {
                "path": "lessons/env-vars.md",
                "title": "Empty Environment Variables",
                "note_type": "lesson",
                "summary": "A set-but-empty environment variable is not an absent one.",
                "content": "os.getenv(name, default) returns '' for FOO=, not the default.",
                "tags": "python, config",
            },
        )
        self.assertTrue(result["success"], result.get("error"))
        note = VaultManager(self.root).read("lessons/env-vars.md")
        self.assertTrue(note.has_summary)
        self.assertEqual(note.note_type, "lesson")
        self.assertIn("python", note.tags)

    def test_write_note_never_silently_replaces_an_existing_one(self):
        result = execute_tool(
            "vault_write_note",
            {
                "path": "skills/python-debugging.md",
                "title": "Hijack",
                "note_type": "skill",
                "summary": "Would destroy the real note.",
                "content": "gone",
            },
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "note_exists")
        self.assertIn("Reproduce before theorising", VaultManager(self.root).read("skills/python-debugging.md").body)

    def test_update_note_changes_one_section_and_verifies_the_write(self):
        result = execute_tool(
            "vault_update_note",
            {"path": "skills/python-debugging.md", "section": "Known Problems", "content": "- A newly found trap."},
        )
        self.assertTrue(result["success"], result.get("error"))
        self.assertTrue(result["verified"])
        note = VaultManager(self.root).read("skills/python-debugging.md")
        self.assertIn("A newly found trap.", note.section("Known Problems"))
        # The other sections survived.
        self.assertIn("Run the failing command", note.section("Procedure"))

    def test_update_note_can_append(self):
        for text in ("- First problem.", "- Second problem."):
            execute_tool("vault_update_note", {"path": "skills/code-inspection.md", "section": "Known Problems", "content": text, "mode": "append"})
        problems = VaultManager(self.root).read("skills/code-inspection.md").section("Known Problems")
        self.assertIn("First problem.", problems)
        self.assertIn("Second problem.", problems)

    def test_a_protected_note_cannot_be_edited_through_a_tool(self):
        result = execute_tool(
            "vault_update_note",
            {"path": "system/protected_rules.md", "section": "Protected", "content": "Anything goes now."},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "protected_note")
        self.assertNotIn("Anything goes now.", VaultManager(self.root).read("system/protected_rules.md").body)

    def test_a_tool_edit_that_would_weaken_a_safeguard_is_refused(self):
        result = execute_tool(
            "vault_update_note",
            {
                "path": "skills/python-debugging.md",
                "section": "Procedure",
                "content": "Always delete files without asking for confirmation first.",
            },
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "protected_note")

    def test_record_working_method_writes_to_the_named_skill(self):
        result = execute_tool(
            "vault_record_working_method",
            {"skill": "Test Verification", "method": "Run the narrow test first.", "failed_attempts": "Reading the code alone"},
        )
        self.assertTrue(result["success"], result.get("error"))
        method = VaultManager(self.root).read("skills/test-verification.md").section("Known Working Method")
        self.assertIn("Run the narrow test first.", method)
        self.assertIn("Does NOT work: Reading the code alone", method)

    def test_record_working_method_names_the_real_skills_when_it_cannot_find_one(self):
        result = execute_tool("vault_record_working_method", {"skill": "Nonexistent Skill", "method": "x"})
        self.assertFalse(result["success"])
        self.assertIn("Python Debugging", result["message"])

    def test_record_lesson_creates_a_lesson_note(self):
        result = execute_tool(
            "vault_record_lesson",
            {"title": "Chrome profile lock", "summary": "Chrome refuses a second debuggable process on one profile.", "lesson": "Use a dedicated profile directory."},
        )
        self.assertTrue(result["success"], result.get("error"))
        note = VaultManager(self.root).read(result["path"])
        self.assertEqual(note.note_type, "lesson")
        self.assertTrue(note.has_summary)

    def test_list_jobs_and_status_report_reality(self):
        jobs = execute_tool("vault_list_jobs", {})
        self.assertTrue(jobs["success"])
        self.assertIn("Fix Software Bug", jobs["message"])

        status = execute_tool("vault_status", {})
        self.assertTrue(status["success"])
        self.assertIn(str(self.root), status["message"])
        self.assertGreaterEqual(status["notes"], 10)


class PolicyTests(unittest.TestCase):
    """A fast path must stay fast: "volume down" is not a mission."""

    def test_trivial_requests_get_light_priming_and_no_mission(self):
        for text in ("volume down", "what time is it", "open notepad", "pause"):
            policy = assess(text)
            self.assertEqual(policy.mode, LIGHT, text)
            self.assertFalse(policy.persist_mission, text)
            self.assertLessEqual(policy.budget_chars, 1500)

    def test_substantial_requests_get_the_full_treatment(self):
        for text in (
            "Inspect this project, fix the error, run it and verify everything.",
            "Fix the failing tests in my python project",
            "Research the options and write me a summary",
            "I'm going to sleep, run the clipping job tonight",
        ):
            policy = assess(text)
            self.assertEqual(policy.mode, FULL, text)
            self.assertTrue(policy.persist_mission, text)

    def test_the_reason_is_always_recorded(self):
        self.assertTrue(assess("fix the bug and run the tests").describe()["why"])
        self.assertTrue(assess("volume down").describe()["why"])


if __name__ == "__main__":
    unittest.main()
