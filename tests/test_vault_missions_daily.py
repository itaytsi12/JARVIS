"""Missions, Daily Notes and startup recovery: Milestones 6, 10 and 11.

The properties under test are about SURVIVAL: a mission whose process
died must still be readable and resumable, and a day's memory must not
depend on a clean shutdown. Every test therefore builds fresh objects
against the same directory rather than reusing the ones that wrote it --
which is exactly what a restart does.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from vault.bootstrap import bootstrap_vault
from vault.daily import DailyJournal
from vault.index import VaultIndex
from vault.manager import VaultManager
from vault.missions import ACTIVE, COMPLETED, FAILED, INTERRUPTED, MissionStore
from vault.startup import recover_session


class VaultTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        self.vault = VaultManager(self.root)
        self.index = VaultIndex(self.vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        bootstrap_vault(self.vault, self.index)
        self.missions = MissionStore(vault=self.vault, index=self.index)
        self.journal = DailyJournal(vault=self.vault, index=self.index)

    def tearDown(self):
        self.temp.cleanup()

    def reopened(self):
        """A completely fresh set of objects on the same directory -- the
        test equivalent of restarting JARVIS."""
        vault = VaultManager(self.root)
        index = VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        return MissionStore(vault=vault, index=index), DailyJournal(vault=vault, index=index)


class MissionPersistenceTests(VaultTestCase):
    def test_a_mission_is_on_disk_before_any_work_happens(self):
        mission = self.missions.create("Fix the failing import in my project", job="Fix Software Bug", skills=["Python Debugging"])
        self.assertTrue(self.vault.note_exists(mission.relative_path))
        self.assertIn("missions/active/", mission.relative_path)
        self.assertEqual(mission.status, ACTIVE)

    def test_progress_is_written_as_it_happens_not_at_the_end(self):
        mission = self.missions.create("Run the tests")
        mission.append_progress("`run_command` succeeded.", step="run_command")
        # Read the FILE, not the object -- a crash right here must not lose it.
        note = self.vault.read(mission.relative_path)
        self.assertIn("run_command", note.section("Progress"))

    def test_a_mission_survives_a_restart_and_can_be_resumed(self):
        mission = self.missions.create("A long overnight job")
        mission.append_progress("Step one done.", step="step_one")
        mission_id = mission.mission_id

        store, _ = self.reopened()
        recovered = store.load(mission_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.goal, "A long overnight job")
        self.assertEqual(recovered.current_step, "step_one")
        self.assertIn("Step one done.", recovered.section("Progress"))

        recovered.resume()
        self.assertEqual(recovered.status, ACTIVE)
        self.assertIn("Mission resumed.", recovered.section("Progress"))

    def test_an_orphaned_mission_is_marked_interrupted_not_guessed_at(self):
        self.missions.create("Something that will be interrupted")
        store, _ = self.reopened()
        orphans = store.mark_orphans_interrupted()
        self.assertEqual(len(orphans), 1)
        self.assertEqual(store.active()[0].status, INTERRUPTED)
        self.assertIn("The JARVIS process stopped", store.active()[0].section("Failures And Retries"))

    def test_a_completed_mission_moves_and_stays_inspectable(self):
        mission = self.missions.create("A short job")
        mission.complete(success=True, outcome="It worked.", verified=True)
        self.assertIn("missions/completed/", mission.relative_path)
        self.assertEqual(mission.status, COMPLETED)
        note = self.vault.read(mission.relative_path)
        self.assertIn("Succeeded (verified)", note.section("Outcome"))
        self.assertEqual(self.missions.active(), [])

    def test_a_failed_mission_is_kept_too(self):
        mission = self.missions.create("A job that fails")
        mission.complete(success=False, outcome="The command never succeeded.")
        self.assertEqual(mission.status, FAILED)
        self.assertIn("missions/completed/", mission.relative_path)

    def test_the_knowledge_used_is_recorded_on_the_mission(self):
        mission = self.missions.create("Fix a bug")
        mission.record_knowledge(
            job="Fix Software Bug",
            skills=["Python Debugging", "Test Verification"],
            notes=["jobs/fix-software-bug.md"],
            rationale="scanned 14 summaries; read 3 in full",
        )
        section = self.vault.read(mission.relative_path).section("Knowledge Loaded")
        self.assertIn("[[Fix Software Bug]]", section)
        self.assertIn("[[Python Debugging]]", section)
        self.assertIn("scanned 14 summaries", section)

    def test_resumable_lists_only_unfinished_missions(self):
        finished = self.missions.create("Finished")
        finished.complete(success=True, outcome="Done.")
        self.missions.create("Unfinished")
        store, _ = self.reopened()
        self.assertEqual([mission.goal for mission in store.resumable()], ["Unfinished"])


class DailyNoteTests(VaultTestCase):
    def test_todays_note_is_created_with_the_full_structure(self):
        today = self.journal.today()
        note = self.vault.read(today.relative_path)
        self.assertEqual(note.note_type, "daily")
        for heading in ("Timeline", "Decisions Made", "User Corrections / Preferences Learned", "Unfinished Work"):
            self.assertIn(heading, note.sections())

    def test_an_event_is_detailed_not_a_one_liner(self):
        today = self.journal.today()
        today.add_event(
            "Fixed the import error",
            request="my project won't start",
            did="Ran the tests, read the traceback, corrected the import.",
            result="The suite passes: 12 passed.",
            files=["brain/router.py"],
            lesson="An empty environment variable is not an absent one.",
        )
        timeline = self.vault.read(today.relative_path).section("Timeline")
        for expected in ("**Asked:**", "**Did:**", "**Result:**", "**Files:**", "**Lesson:**", "brain/router.py"):
            self.assertIn(expected, timeline)

    def test_appends_accumulate_and_survive_without_any_shutdown(self):
        today = self.journal.today()
        today.add_event("First piece of work")
        today.add_event("Second piece of work")
        today.add_problem("A command failed once.")

        _, journal = self.reopened()
        note = self.vault.read(journal.today().relative_path)
        self.assertIn("First piece of work", note.section("Timeline"))
        self.assertIn("Second piece of work", note.section("Timeline"))
        self.assertIn("A command failed once.", note.section("Problems Encountered"))

    def test_an_identical_line_is_not_duplicated(self):
        today = self.journal.today()
        for _ in range(3):
            today.add_problem("The same problem.")
        problems = self.vault.read(today.relative_path).section("Problems Encountered")
        self.assertEqual(problems.count("The same problem."), 1)

    def test_the_quick_summary_is_derived_from_what_actually_happened(self):
        today = self.journal.today()
        today.add_event("Did a thing")
        today.add_correction("Always reuse an open window.")
        today.refresh_quick_summary()
        note = self.vault.read(today.relative_path)
        self.assertIn("1 recorded pieces of work today.", note.quick_summary)
        self.assertIn("Corrections learned: 1.", note.quick_summary)
        self.assertIn("user corrections", note.summary)

    def test_a_credential_is_never_written_to_the_daily_note(self):
        today = self.journal.today()
        today.add_event("Configured a service", did="Set api_key=sk-abcdefghijklmnopqrstuvwxyz012345 in the file")
        timeline = self.vault.read(today.relative_path).section("Timeline")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz012345", timeline)
        self.assertIn("<REDACTED>", timeline)

    def test_the_previous_day_is_the_most_recent_one_with_a_note(self):
        """Literally-yesterday is often wrong: JARVIS is not used daily."""
        stamp = (datetime.now().date() - timedelta(days=4)).strftime("%Y-%m-%d")
        older = self.journal.for_date(stamp)
        older.add_event("Work from several days ago")
        found = self.journal.yesterday()
        self.assertIsNotNone(found)
        self.assertEqual(found.date, stamp)


class StartupRecoveryTests(VaultTestCase):
    def test_a_new_session_recovers_preferences_missions_and_the_recent_days(self):
        mission = self.missions.create("Something left running")
        mission.append_progress("Got partway.", step="halfway")
        yesterday = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.journal.for_date(yesterday).add_event("Yesterday's work", result="Left the import half fixed.")

        vault = VaultManager(self.root)
        index = VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        recovery = recover_session(
            vault=vault,
            index=index,
            journal=DailyJournal(vault=vault, index=index),
            missions=MissionStore(vault=vault, index=index),
        )

        self.assertGreater(recovery.notes, 10)
        self.assertTrue(recovery.preferences)
        self.assertEqual(len(recovery.resumable_missions), 1)
        self.assertEqual(recovery.interrupted, [mission.relative_path])
        self.assertIn("Yesterday", recovery.previous_day or "")

        text = recovery.context_text()
        self.assertIn("What the user prefers", text)
        self.assertIn("Missions left unfinished", text)
        self.assertIn("Something left running", text)

    def test_recovery_never_raises_when_the_vault_is_unreadable(self):
        broken = VaultManager(Path(self.temp.name) / "does-not-exist" / "nested")
        recovery = recover_session(vault=broken, index=VaultIndex(broken, cache_path=None, use_cache=False))
        self.assertIsInstance(recovery.describe(), dict)

    def test_the_spoken_summary_mentions_an_unfinished_mission_and_nothing_otherwise(self):
        vault = VaultManager(self.root)
        index = VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        quiet = recover_session(vault=vault, index=index, journal=DailyJournal(vault=vault, index=index), missions=MissionStore(vault=vault, index=index))
        self.assertEqual(quiet.spoken_summary(), "")

        self.missions.create("An unfinished mission")
        loud = recover_session(vault=vault, index=index, journal=DailyJournal(vault=vault, index=index), missions=MissionStore(vault=vault, index=index))
        self.assertIn("unfinished mission", loud.spoken_summary())


if __name__ == "__main__":
    unittest.main()
