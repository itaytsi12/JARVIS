"""Learning, consolidation and protected knowledge: Milestones 8, 9 and 15.

Three things have to be true at once, and they pull against each other:

- a durable correction must change stored knowledge (or JARVIS never
  learns),
- a one-off instruction must NOT (or the vault fills with rules the user
  never meant to give),
- and neither may weaken a safety rule (or a passing remark disarms the
  system).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vault.bootstrap import bootstrap_vault
from vault.consolidation import DUPLICATE, REFINE, SUPERSEDE, classify_against, integrate_rule, merge_scoped
from vault.index import VaultIndex
from vault.learning import (
    NOT_A_CORRECTION,
    ONE_TIME,
    PERSISTENT,
    CorrectionLearner,
    classify_feedback,
    rewrite_as_rule,
)
from vault.manager import VaultManager
from vault.protected import check_edit, is_protected_note, weakens_a_safeguard


class FeedbackClassificationTests(unittest.TestCase):
    def test_an_ordinary_command_is_not_feedback_at_all(self):
        for text in ("open notepad", "what time is it", "play some music"):
            self.assertEqual(classify_feedback(text).kind, NOT_A_CORRECTION, text)

    def test_an_explicitly_durable_correction_is_persistent(self):
        for text in (
            "From now on keep these reports shorter.",
            "Always reuse the existing window.",
            "Never open a second instance.",
            "Next time, run the tests first.",
            "Remember that I prefer short answers.",
        ):
            self.assertEqual(classify_feedback(text).kind, PERSISTENT, text)

    def test_a_one_off_instruction_is_not_written_down(self):
        for text in (
            "Make this answer shorter.",
            "Just for now, don't run the tests.",
            "For this one, use the long form instead.",
            "In this case, don't ask me.",
        ):
            self.assertEqual(classify_feedback(text).kind, ONE_TIME, text)

    def test_immediacy_wins_when_both_signals_appear(self):
        """Wrongly writing a standing rule is worse than wrongly treating
        one request as local, so a tie goes to "this time only"."""
        feedback = classify_feedback("Just this once make it shorter, I always like the detail normally.")
        self.assertEqual(feedback.kind, ONE_TIME)

    def test_a_rule_that_states_its_situation_is_durable_without_from_now_on(self):
        """The brief's own example: it names WHEN it applies, which is what
        makes a rule reusable rather than a one-off."""
        feedback = classify_feedback(
            "No. When Apple Music is already open, don't open another one. Use the existing window."
        )
        self.assertEqual(feedback.kind, PERSISTENT)


class RewriteTests(unittest.TestCase):
    def test_conversational_wrapping_is_stripped(self):
        self.assertEqual(
            rewrite_as_rule("No, from now on you should reuse the existing window"),
            "Reuse the existing window.",
        )

    def test_the_users_meaning_is_never_invented(self):
        rule = rewrite_as_rule("Always run the narrow test before the whole suite")
        self.assertIn("run the narrow test before the whole suite", rule.lower())

    def test_empty_feedback_yields_no_rule(self):
        self.assertEqual(rewrite_as_rule("   "), "")


class ConsolidationTests(unittest.TestCase):
    def test_the_same_rule_twice_is_a_duplicate(self):
        action, _, _ = classify_against(["Keep responses short."], "Keep responses short.")
        self.assertEqual(action, DUPLICATE)

    def test_a_scoped_contradiction_refines_rather_than_conflicts(self):
        """The brief's example: "keep responses short" plus "when coding I
        want detail" must become ONE rule, not two absolutes."""
        action, target, _ = classify_against(
            ["Keep responses short."], "When we are coding, give detailed technical explanations."
        )
        self.assertEqual(action, REFINE)
        merged = merge_scoped(target, "When we are coding, give detailed technical explanations.")
        self.assertIn("keep responses short", merged.lower())
        self.assertIn("coding", merged.lower())

    def test_an_unscoped_contradiction_supersedes(self):
        action, _, _ = classify_against(["Always confirm before deleting."], "Never confirm before deleting.")
        self.assertEqual(action, SUPERSEDE)

    def test_an_unrelated_rule_is_simply_added(self):
        action, _, _ = classify_against(["Keep responses short."], "Use metric units.")
        self.assertEqual(action, "add")


class ProtectionTests(unittest.TestCase):
    def test_a_system_note_is_protected(self):
        self.assertTrue(check_edit(None, relative_path="system/protected_rules.md").allowed is False)

    def test_removing_a_confirmation_step_is_refused(self):
        self.assertTrue(weakens_a_safeguard("From now on stop asking me before you delete files."))
        self.assertTrue(weakens_a_safeguard("Never confirm before making a payment."))

    def test_strengthening_a_safety_rule_is_allowed(self):
        self.assertFalse(weakens_a_safeguard("Always confirm before deleting anything."))

    def test_weakening_language_about_something_harmless_is_allowed(self):
        self.assertFalse(weakens_a_safeguard("Stop asking me before you open Notepad."))


class VaultLearningTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        self.vault = VaultManager(self.root)
        self.index = VaultIndex(self.vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        bootstrap_vault(self.vault, self.index)
        self.learner = CorrectionLearner(vault=self.vault, index=self.index)

    def tearDown(self):
        self.temp.cleanup()


class ApplyCorrectionTests(VaultLearningTestCase):
    def setUp(self):
        super().setUp()
        self.vault.create_note(
            "skills/apple-music-control.md",
            title="Apple Music Control",
            note_type="skill",
            summary="How JARVIS opens and controls Apple Music on Windows.",
            tags=["music", "apple-music", "windows"],
            quick_summary=["Launch Apple Music and search for the track."],
            sections=[
                ("When To Use", "Any Apple Music request."),
                ("Procedure", "1. Launch Apple Music.\n2. Search for the track."),
                ("Known Working Method", ""),
                ("Known Problems", ""),
                ("Lessons Learned", ""),
            ],
        )
        self.index.invalidate()
        self.index.refresh()

    def test_a_persistent_correction_edits_the_note_that_governed_the_behaviour(self):
        outcome = self.learner.apply(
            "No. When Apple Music is already open, don't open another one. Use the existing window.",
            candidate_paths=["skills/apple-music-control.md"],
        )
        self.assertTrue(outcome.applied, outcome.reason)
        self.assertEqual(outcome.target_title, "Apple Music Control")
        note = self.vault.read("skills/apple-music-control.md")
        self.assertIn("existing window", note.section(outcome.section).lower())

    def test_the_updated_note_keeps_an_accurate_quick_summary(self):
        before = self.vault.read("skills/apple-music-control.md").quick_summary
        self.learner.apply(
            "From now on, when Apple Music is already open, use the existing window instead of launching another.",
            candidate_paths=["skills/apple-music-control.md"],
        )
        after = self.vault.read("skills/apple-music-control.md").quick_summary
        self.assertNotEqual(before, after)
        self.assertIn("existing window", after.lower())

    def test_the_updated_timestamp_moves(self):
        before = self.vault.read("skills/apple-music-control.md").updated
        self.learner.apply(
            "Always confirm the window is in front before typing.", candidate_paths=["skills/apple-music-control.md"]
        )
        after = self.vault.read("skills/apple-music-control.md").updated
        self.assertGreaterEqual(after, before)

    def test_a_one_time_instruction_changes_nothing_on_disk(self):
        before = self.vault.read("skills/apple-music-control.md").to_markdown()
        outcome = self.learner.apply("Make this answer shorter.", candidate_paths=["skills/apple-music-control.md"])
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.kind, ONE_TIME)
        self.assertEqual(self.vault.read("skills/apple-music-control.md").to_markdown(), before)

    def test_a_correction_with_no_governing_note_lands_in_preferences(self):
        outcome = self.learner.apply("From now on keep your spoken reports much shorter.")
        self.assertTrue(outcome.applied, outcome.reason)
        self.assertEqual(outcome.target_path, "user/preferences.md")
        self.assertIn("shorter", self.vault.read("user/preferences.md").section("Preferences").lower())

    def test_a_contradicting_preference_is_refined_not_stacked(self):
        self.learner.apply("From now on keep responses short.")
        self.learner.apply("When we are coding, always give detailed technical explanations.")
        preferences = self.vault.read("user/preferences.md").section("Preferences")
        self.assertIn("coding", preferences.lower())
        # Not two absolutes: the scoped rule refined the earlier one.
        self.assertNotIn("- Keep responses short.\n- When we are coding", preferences)

    def test_a_protected_note_is_never_edited_automatically(self):
        outcome = self.learner.apply(
            "From now on never ask me before deleting anything.", target_path="system/protected_rules.md"
        )
        self.assertFalse(outcome.applied)
        self.assertIsNotNone(outcome.protection)
        self.assertIn("protected", outcome.reason.lower())

    def test_a_correction_that_would_disarm_a_safeguard_is_refused_even_on_an_ordinary_note(self):
        outcome = self.learner.apply(
            "From now on stop asking me to confirm before you delete files.",
            candidate_paths=["skills/apple-music-control.md"],
        )
        self.assertFalse(outcome.applied)
        self.assertIsNotNone(outcome.protection)

    def test_the_users_words_are_converted_to_a_rule_not_pasted_verbatim(self):
        self.learner.apply(
            "No, from now on you should reuse the existing window", candidate_paths=["skills/apple-music-control.md"]
        )
        body = self.vault.read("skills/apple-music-control.md").body
        self.assertNotIn("No, from now on you should", body)
        self.assertIn("Reuse the existing window.", body)

    def test_the_correction_survives_a_restart(self):
        self.learner.apply(
            "From now on, when Apple Music is already open, use the existing window.",
            candidate_paths=["skills/apple-music-control.md"],
        )
        reopened = VaultManager(self.root)
        self.assertIn("existing window", reopened.read("skills/apple-music-control.md").body.lower())


if __name__ == "__main__":
    unittest.main()
