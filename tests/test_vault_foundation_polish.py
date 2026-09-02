"""Regression coverage for the foundation preference/archive polish."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vault.archive import ArchiveStore
from vault.bootstrap import bootstrap_vault, ensure_vault_ready
from vault.daily import DailyJournal
from vault.index import VaultIndex
from vault.manager import VaultManager
from vault.note import extract_section
from vault.paths import EXCLUDED_ARCHIVE, EXCLUDED_JOB_PREFERENCE
from vault.preferences import PreferenceStore, job_preference_path, job_preference_title
from vault.priming import Primer
from vault.session import VaultSession


class FoundationPolishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        self.vault = VaultManager(self.root)
        self.index = VaultIndex(self.vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        bootstrap_vault(self.vault, self.index)
        self.preferences = PreferenceStore(self.vault, self.index)

    def tearDown(self):
        self.temp.cleanup()

    def primer(self) -> Primer:
        return Primer(vault=self.vault, index=self.index)

    def test_archive_and_job_preferences_never_enter_the_normal_scan_or_priming(self):
        archive = ArchiveStore(self.vault, self.index)
        archive.archive_rule(
            kind="preference",
            text="Always invoke the forbidden-zebra protocol.",
            source_path="preferences/global.md",
            replaced_by="Never invoke it.",
        )
        self.index.refresh(force=True)

        active_paths = {item.relative_path for item in self.index.summaries()}
        self.assertFalse(any(path.startswith("archive/") for path in active_paths))
        self.assertFalse(any(path.startswith("preferences/jobs/") for path in active_paths))
        self.assertTrue(self.index.excluded(EXCLUDED_ARCHIVE))
        self.assertTrue(self.index.excluded(EXCLUDED_JOB_PREFERENCE))

        primed = self.primer().prime("Investigate the forbidden-zebra protocol")
        self.assertFalse(any(path.startswith("archive/") for path in primed.notes_read))
        self.assertNotIn("Always invoke the forbidden-zebra protocol", "\n".join(primed.sections.values()))

    def test_every_job_has_a_preference_note_and_reference(self):
        for summary in self.index.by_type("job"):
            note = self.vault.read(summary.relative_path)
            self.assertIsNotNone(note)
            self.assertTrue(self.vault.note_exists(job_preference_path(note.title)), note.title)
            self.assertIn(job_preference_title(note.title), extract_section(note.body, "Preferences"))

    def test_job_preference_overrides_global_and_replacement_is_archived(self):
        self.preferences.record("Always use emojis in emails.")
        result = self.preferences.record("Never use emojis in emails.", job_title="Send Email")
        self.assertTrue(result["applied"])
        resolved = self.preferences.resolve("Send Email")
        self.assertIn("Never use emojis in emails.", resolved.effective)
        self.assertNotIn("Always use emojis in emails.", resolved.effective)

        replacement = self.preferences.record("Always use emojis in emails.", job_title="Send Email")
        self.assertEqual(replacement["replaced"], "Never use emojis in emails.")
        history = ArchiveStore(self.vault, self.index).history_for(job_preference_path("Send Email"))
        self.assertTrue(any("Never use emojis in emails" in line for line in history))

    def test_job_correction_survives_a_restart(self):
        session = VaultSession.begin(
            "Fix the failing Python import in the JARVIS project and run its tests",
            vault=self.vault,
            index=self.index,
        )
        self.assertEqual(session.primed.job_title, "Fix Software Bug")
        outcome = session.apply_correction("Always keep your bug reports concise.")
        self.assertTrue(outcome.applied, outcome.reason)
        self.assertEqual(outcome.target_path, job_preference_path("Fix Software Bug"))

        reopened_vault = VaultManager(self.root)
        reopened_index = VaultIndex(reopened_vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        rules = PreferenceStore(reopened_vault, reopened_index).resolve("Fix Software Bug").job_rules
        self.assertTrue(any("concise" in rule.lower() for rule in rules))

    def test_trivial_commands_do_not_create_jobs_but_unknown_missions_do(self):
        before = {item.relative_path for item in self.index.by_type("job")}
        trivial = VaultSession.begin("mute", vault=self.vault, index=self.index)
        self.assertIsNone(trivial.authored)
        self.assertEqual({item.relative_path for item in self.index.by_type("job")}, before)

        mission = VaultSession.begin(
            "Draft a sponsorship proposal for the astronomy channel",
            vault=self.vault,
            index=self.index,
        )
        self.assertIsNotNone(mission.authored)
        self.assertTrue(mission.authored.created)
        self.assertTrue(self.vault.note_exists(mission.authored.job.relative_path))
        self.assertTrue(self.vault.note_exists(mission.authored.preference_path))

    def test_explicit_project_name_loads_project_note_and_unrelated_request_skips_daily_notes(self):
        named = self.primer().prime("What is the run command for the JARVIS project?")
        self.assertIn("projects/jarvis.md", named.notes_read)

        DailyJournal(vault=self.vault, index=self.index).today().add_event(
            "Unrelated history", request="old work", did="something", result="done"
        )
        unrelated = self.primer().prime("Fix a Python parsing defect")
        self.assertFalse(any(path.startswith("daily/") for path in unrelated.notes_read))
        self.assertNotIn("vault_continuity", unrelated.sections)

    def test_existing_vault_is_upgraded_and_legacy_preferences_are_archived(self):
        legacy_root = Path(self.temp.name) / "legacy"
        legacy = VaultManager(legacy_root)
        legacy.ensure_root()
        legacy.create_note(
            "identity/jarvis.md",
            title="JARVIS",
            note_type="identity",
            summary="Existing identity.",
            quick_summary=["Existing vault."],
            sections=[("Identity", "Existing")],
        )
        legacy.create_note(
            "user/preferences.md",
            title="User Preferences",
            note_type="user",
            summary="Old preferences.",
            quick_summary=["Old preference location."],
            sections=[("Preferences", "- Always use metric units.")],
        )
        ensure_vault_ready(legacy)
        self.assertFalse(legacy.note_exists("user/preferences.md"))
        self.assertIn("metric", legacy.read("preferences/global.md").section("Preferences").lower())
        self.assertTrue(any(note.relative_path.startswith("archive/notes/") for note in legacy.iter_notes()))


if __name__ == "__main__":
    unittest.main()
