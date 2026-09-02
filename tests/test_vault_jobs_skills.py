"""Jobs and Skills, discovered from the vault: Milestones 4, 5 and 12.

The headline requirement is that adding `jobs/new_job.md` makes a new Job
available WITHOUT touching any routing code. These tests prove it by
writing a note to disk and then asking the registry for it -- nothing is
registered, imported or patched in between.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vault.bootstrap import bootstrap_vault
from vault.index import VaultIndex
from vault.jobs import JobRegistry
from vault.manager import VaultManager
from vault.skills import SkillLibrary


class VaultTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault = VaultManager(Path(self.temp.name) / "vault")
        self.index = VaultIndex(self.vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        bootstrap_vault(self.vault, self.index)
        self.jobs = JobRegistry(index=self.index, vault=self.vault)
        self.skills = SkillLibrary(index=self.index, vault=self.vault)

    def tearDown(self):
        self.temp.cleanup()


class JobDiscoveryTests(VaultTestCase):
    def test_the_seeded_jobs_are_discovered(self):
        titles = self.jobs.titles()
        self.assertIn("Fix Software Bug", titles)
        self.assertIn("Answer About This Machine", titles)

    def test_a_job_is_selected_from_its_summary_and_when_to_use(self):
        job = self.jobs.select("my python tests are failing, fix the bug")
        self.assertIsNotNone(job)
        self.assertEqual(job.title, "Fix Software Bug")

    def test_no_job_is_forced_onto_an_unrelated_request(self):
        self.assertIsNone(self.jobs.select("what is the capital of France"))

    def test_a_new_job_needs_no_code_change_at_all(self):
        """The requirement in full: drop a Markdown file into jobs/ and it
        is selectable. Nothing below registers, imports or patches."""
        self.vault.create_note(
            "jobs/write-sales-email.md",
            title="Write Sales Email",
            note_type="job",
            summary="Draft a persuasive outbound sales email for a named prospect.",
            tags=["job", "sales", "email", "writing"],
            quick_summary=["Use when the user asks for a sales or outreach email."],
            sections=[
                ("When To Use", "The user asks for a sales email, outreach message or cold email."),
                ("Required Skills", "- [[Code Inspection]]"),
                ("Procedure", "1. Research the prospect.\n2. Draft.\n3. Tighten."),
            ],
        )
        self.index.invalidate()
        self.index.refresh()

        job = self.jobs.select("write me a sales email for that prospect")
        self.assertIsNotNone(job)
        self.assertEqual(job.title, "Write Sales Email")
        self.assertIn("Research the prospect", job.procedure)

    def test_a_placeholder_job_is_never_selected_automatically(self):
        """The Clipping Job describes work that is not implemented. It has
        to be visible and NOT selectable, or JARVIS would confidently
        attempt something with no procedure behind it."""
        self.assertIn("Clipping", self.jobs.titles())
        for job in self.jobs.rank("run the clipping job tonight"):
            if job.title == "Clipping":
                self.assertFalse(job.selectable)
        selected = self.jobs.select("run the clipping job tonight")
        self.assertNotEqual(getattr(selected, "title", None), "Clipping")

    def test_the_catalog_lists_every_job_in_one_line_each(self):
        catalog = self.jobs.catalog()
        self.assertIn("Fix Software Bug:", catalog)
        self.assertIn("[placeholder]", catalog)

    def test_a_job_declares_its_skills_as_wikilinks(self):
        job = self.jobs.load("Fix Software Bug")
        self.assertEqual(job.required_skills, ["Code Inspection", "Python Debugging", "Test Verification"])

    def test_job_guidance_carries_the_procedure_but_not_the_selection_criteria(self):
        job = self.jobs.load("Fix Software Bug")
        guidance = job.guidance()
        self.assertIn("Reproduce: run the failing command", guidance)
        self.assertNotIn("When To Use", guidance)


class SkillTests(VaultTestCase):
    def test_a_jobs_skills_are_loaded_by_title(self):
        job = self.jobs.load("Fix Software Bug")
        loaded, missing = self.skills.load_all(job.required_skills)
        self.assertEqual(missing, [])
        self.assertEqual({skill.title for skill in loaded}, {"Code Inspection", "Python Debugging", "Test Verification"})

    def test_a_named_but_missing_skill_is_reported_not_hidden(self):
        job = self.jobs.load("Clipping")
        loaded, missing = self.skills.load_all(job.required_skills)
        self.assertEqual(loaded, [])
        self.assertIn("Campaign Discovery _(not built)_", missing)

    def test_one_skill_supports_several_jobs(self):
        bug = self.jobs.load("Fix Software Bug")
        machine = self.jobs.load("Answer About This Machine")
        self.assertIn("Code Inspection", bug.required_skills)
        self.assertIn("Code Inspection", machine.required_skills)

    def test_a_working_method_is_recorded_and_survives_a_reload(self):
        """Milestone 12: do not pay the discovery tax twice."""
        updated = self.skills.record_working_method(
            "Python Debugging",
            method="Run the narrow failing test first; the suite afterwards.",
            failed_attempts=["Reading the code without running it"],
            source="mission abc123",
        )
        self.assertIsNotNone(updated)
        reopened = SkillLibrary(index=VaultIndex(self.vault, cache_path=None, use_cache=False), vault=VaultManager(self.vault.root))
        method = reopened.load("Python Debugging").known_working_method
        self.assertIn("Run the narrow failing test first", method)
        self.assertIn("Does NOT work: Reading the code without running it", method)

    def test_the_same_working_method_is_not_recorded_twice(self):
        for _ in range(3):
            self.skills.record_working_method("Python Debugging", method="Always read the real traceback.")
        method = self.skills.load("Python Debugging").known_working_method
        self.assertEqual(method.count("Always read the real traceback."), 1)

    def test_a_failed_approach_is_recorded(self):
        self.skills.record_failed_approach(
            "Windows Desktop Control", approach="Clicking by screen coordinates", reason="breaks when a window moves"
        )
        problems = self.skills.load("Windows Desktop Control").section("Known Problems")
        self.assertIn("Clicking by screen coordinates -- breaks when a window moves", problems)

    def test_recording_against_a_missing_skill_reports_it(self):
        self.assertIsNone(self.skills.record_working_method("No Such Skill", method="x"))

    def test_skill_guidance_puts_the_working_method_first(self):
        self.skills.record_working_method("Test Verification", method="Re-run the exact failing command.")
        guidance = self.skills.load("Test Verification").guidance()
        self.assertLess(guidance.index("Known Working Method"), guidance.index("Procedure"))


if __name__ == "__main__":
    unittest.main()
