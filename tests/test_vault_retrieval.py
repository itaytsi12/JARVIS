"""Two-stage retrieval and the index: Milestone 3.

The claim being tested is specific and measurable: JARVIS can triage
hundreds of notes without reading them, and only the relevant ones enter
the model's context. A test that merely asserted "the right note came
back" would pass on a system that read the whole vault every time, so the
tests here check the COST as well as the answer.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vault.bootstrap import bootstrap_vault
from vault.index import VaultIndex
from vault.manager import VaultManager
from vault.retrieval import VaultRetriever


class VaultTestCase(unittest.TestCase):
    """A real vault in a temporary directory. No mocks: the thing under
    test is the interaction with real files."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        self.vault = VaultManager(self.root)
        self.index = VaultIndex(self.vault, cache_path=Path(self.temp.name) / "cache.json", refresh_interval=0.0)
        bootstrap_vault(self.vault, self.index)

    def tearDown(self):
        self.temp.cleanup()

    def add_note(self, path, title, note_type, summary, tags=(), quick=(), body=""):
        return self.vault.create_note(
            path,
            title=title,
            note_type=note_type,
            summary=summary,
            tags=tags,
            quick_summary=quick or [summary],
            sections=[("Detail", body or "Detail.")],
        )


class IndexTests(VaultTestCase):
    def test_every_seeded_note_is_indexed_with_a_summary(self):
        summaries = self.index.refresh()
        self.assertGreaterEqual(len(summaries), 10)
        for item in summaries:
            self.assertTrue(item.summary, f"{item.relative_path} has no summary")
            self.assertTrue(item.title)
            self.assertTrue(item.note_type)

    def test_a_changed_note_is_re_read_and_an_unchanged_one_is_not(self):
        self.index.refresh(force=True)
        self.index.refresh()
        self.assertEqual(self.index.last_reparsed, 0)
        self.add_note("skills/new.md", "New Skill", "skill", "A newly added skill.")
        self.index.invalidate()
        self.index.refresh()
        self.assertEqual(self.index.last_reparsed, 1)

    def test_a_deleted_note_leaves_the_index(self):
        self.add_note("lessons/temporary.md", "Temporary", "lesson", "Will be removed.")
        self.index.invalidate()
        self.index.refresh()
        self.assertIsNotNone(self.index.get("lessons/temporary.md"))
        (self.root / "lessons" / "temporary.md").unlink()
        self.index.invalidate()
        self.index.refresh()
        self.assertIsNone(self.index.get("lessons/temporary.md"))

    def test_the_cache_survives_a_new_index_object(self):
        self.index.refresh(force=True)
        rebuilt = VaultIndex(self.vault, cache_path=self.index.cache_path, refresh_interval=999.0)
        self.assertGreaterEqual(len(rebuilt.summaries(refresh=False)), 10)

    def test_a_cache_from_a_different_vault_is_ignored(self):
        self.index.refresh(force=True)
        other = VaultManager(self.root.parent / "other-vault")
        other.ensure_root()
        rebuilt = VaultIndex(other, cache_path=self.index.cache_path, refresh_interval=999.0)
        self.assertEqual(rebuilt.summaries(refresh=False), [])

    def test_the_markdown_index_is_written_and_openable(self):
        path = self.index.write_markdown_index()
        self.assertTrue(path.is_file())
        note = self.vault.read("VAULT_INDEX.md")
        self.assertEqual(note.note_type, "index")
        self.assertTrue(note.has_summary)
        self.assertIn("Fix Software Bug", note.body)

    def test_generated_index_notes_are_never_retrieval_candidates(self):
        self.index.write_markdown_index()
        self.index.invalidate()
        candidates, _, _ = VaultRetriever(index=self.index, vault=self.vault).scan("index of jobs and skills")
        self.assertEqual([item for item in candidates if item.summary.note_type == "index"], [])


class TwoStageRetrievalTests(VaultTestCase):
    def setUp(self):
        super().setUp()
        self.add_note(
            "skills/apple-music-control.md",
            "Apple Music Control",
            "skill",
            "How JARVIS opens, reuses, searches and controls Apple Music on Windows.",
            tags=["music", "apple-music"],
            quick=["Reuse an open window rather than launching a second one."],
            body="x" * 3000,
        )
        self.add_note(
            "skills/spotify-control.md",
            "Spotify Control",
            "skill",
            "How JARVIS controls Spotify playback.",
            tags=["music", "spotify"],
            body="y" * 3000,
        )
        self.add_note(
            "skills/video-editing.md",
            "Video Editing",
            "skill",
            "Short-form video processing procedures with FFmpeg.",
            tags=["video"],
            body="z" * 3000,
        )
        self.index.invalidate()
        self.index.refresh()
        self.retriever = VaultRetriever(index=self.index, vault=self.vault)

    def test_the_relevant_note_is_found_through_its_summary(self):
        result = self.retriever.retrieve("Fix the Apple Music playlist problem.", limit=3)
        self.assertIn("skills/apple-music-control.md", result.paths())

    def test_an_unrelated_note_is_never_deep_read(self):
        result = self.retriever.retrieve("Fix the Apple Music playlist problem.", limit=3)
        self.assertNotIn("skills/video-editing.md", result.paths())

    def test_the_scan_sees_everything_and_the_read_sees_almost_nothing(self):
        result = self.retriever.retrieve("Fix the Apple Music playlist problem.", limit=3)
        trace = result.trace
        self.assertGreaterEqual(trace.scanned, 13)
        self.assertLessEqual(len(result.notes), 6)
        full_vault_chars = sum(item.size for item in self.index.summaries())
        self.assertLess(trace.deep_read_chars, full_vault_chars * 0.5)

    def test_a_structural_bonus_alone_never_qualifies_a_note(self):
        """A recently-touched note of the requested TYPE must not be
        selected on that basis. Both bonuses together came to 1.15 and
        cleared a 1.0 threshold, which loaded a video-editing skill into
        an Apple Music mission."""
        candidates, _, _ = self.retriever.scan("xyzzy nothing matches this", types=["skill"])
        self.assertEqual(candidates, [])

    def test_the_trace_says_what_was_considered_and_what_was_chosen(self):
        result = self.retriever.retrieve("Fix the Apple Music playlist problem.", limit=3)
        explanation = result.trace.explain()
        self.assertIn("Candidates considered", explanation)
        self.assertIn("Apple Music Control", explanation)
        self.assertIn("Selected for full read", explanation)
        described = result.trace.describe()
        self.assertTrue(described["considered"])
        self.assertTrue(described["selected"])

    def test_the_budget_is_enforced(self):
        result = self.retriever.retrieve("music", limit=8, budget_chars=1200)
        self.assertLessEqual(result.trace.deep_read_chars, 1200)

    def test_a_linked_note_is_pulled_in_one_hop(self):
        self.add_note(
            "jobs/tidy-playlists.md",
            "Tidy Playlists",
            "job",
            "Reorganise the user's music playlists.",
            tags=["playlist"],
            body="## Required Skills\n\n- [[Apple Music Control]]\n",
        )
        self.index.invalidate()
        self.index.refresh()
        result = self.retriever.retrieve("Tidy Playlists", limit=1)
        self.assertIn("jobs/tidy-playlists.md", result.paths())
        self.assertIn("skills/apple-music-control.md", result.paths())

    def test_always_notes_are_read_regardless_of_score(self):
        result = self.retriever.retrieve("something with no overlap at all", always=["identity/core_rules.md"])
        self.assertIn("identity/core_rules.md", result.paths())


class ScaleTests(VaultTestCase):
    """Hundreds of note summaries can be scanned without loading their
    bodies -- the specific claim the two-stage design is built on."""

    def setUp(self):
        super().setUp()
        for number in range(300):
            self.add_note(
                f"lessons/decoy-{number:03d}.md",
                f"Decoy Lesson {number}",
                "lesson",
                f"An unrelated observation number {number} about gardening.",
                tags=["decoy"],
                body="q" * 1200,
            )
        self.add_note(
            "skills/needle.md",
            "Kubernetes Deployment",
            "skill",
            "How JARVIS deploys a service to a Kubernetes cluster.",
            tags=["kubernetes", "deployment"],
            body="k" * 2000,
        )
        self.index.invalidate()
        self.index.refresh()

    def test_the_needle_is_found_among_three_hundred_summaries(self):
        retriever = VaultRetriever(index=self.index, vault=self.vault)
        result = retriever.retrieve("deploy the service to kubernetes", limit=3)
        self.assertGreaterEqual(result.trace.scanned, 300)
        self.assertIn("skills/needle.md", result.paths())

    def test_scanning_costs_a_small_fraction_of_reading(self):
        statistics = self.index.statistics()
        self.assertGreaterEqual(statistics["notes"], 300)
        self.assertLess(statistics["scan_fraction_of_full"], 0.35)

    def test_the_decoys_are_seen_but_not_read(self):
        retriever = VaultRetriever(index=self.index, vault=self.vault)
        result = retriever.retrieve("deploy the service to kubernetes", limit=3)
        self.assertFalse([path for path in result.paths() if "decoy" in path])


if __name__ == "__main__":
    unittest.main()
