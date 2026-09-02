"""The note format and the VaultManager: Milestones 1 and 2.

The properties that matter here are the ones the rest of the system
assumes without checking: a note JARVIS wrote can be read back byte for
byte, a note a human broke does not take down a scan, and there is no way
to create a knowledge note without a summary.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vault.manager import OutsideVault, VaultError, VaultManager
from vault.note import (
    Note,
    build_note_text,
    dump_frontmatter,
    extract_list_items,
    extract_quick_summary,
    extract_section,
    extract_wikilinks,
    parse_frontmatter,
    replace_section,
)


class FrontmatterTests(unittest.TestCase):
    def test_parses_a_well_formed_note(self):
        text = (
            "---\ntitle: Apple Music Control\ntype: skill\n"
            "summary: How JARVIS controls Apple Music.\ntags:\n  - music\n  - windows\n"
            "updated: 2026-09-02T14:03:11+00:00\n---\n\n# Apple Music Control\n\nBody.\n"
        )
        metadata, body = parse_frontmatter(text)
        self.assertEqual(metadata["type"], "skill")
        self.assertEqual(metadata["tags"], ["music", "windows"])
        self.assertIn("# Apple Music Control", body)

    def test_a_timestamp_stays_a_string(self):
        """PyYAML turns an ISO timestamp into a datetime, whose str() is
        space-separated -- so a note merely read and written back came out
        byte-different from the one JARVIS had just written."""
        metadata, _ = parse_frontmatter("---\nupdated: 2026-09-02T14:03:11+00:00\n---\n\nx\n")
        self.assertIsInstance(metadata["updated"], str)
        self.assertEqual(metadata["updated"], "2026-09-02T14:03:11+00:00")

    def test_a_note_with_no_frontmatter_keeps_its_body(self):
        metadata, body = parse_frontmatter("# Just a heading\n\nSome text.\n")
        self.assertEqual(metadata, {})
        self.assertIn("Just a heading", body)

    def test_broken_frontmatter_never_raises_and_never_loses_the_body(self):
        text = "---\nthis: is: not: valid\n\tand a tab\n---\n\n# Real content\n\nThe user's words.\n"
        metadata, body = parse_frontmatter(text)
        self.assertIn("The user's words.", body)
        self.assertIsInstance(metadata, dict)

    def test_round_trip_is_byte_stable(self):
        text = build_note_text(
            title="Round Trip",
            note_type="skill",
            summary="Proves a note survives being read and written back.",
            tags=["a", "b"],
            quick_summary=["One.", "Two."],
            sections=[("Procedure", "1. Do the thing.")],
        )
        note = Note.from_text(text, path=Path("x.md"), relative_path="skills/x.md")
        self.assertEqual(note.to_markdown(), text)

    def test_an_unknown_frontmatter_field_is_preserved(self):
        rendered = dump_frontmatter({"title": "T", "type": "skill", "obsidian_cssclass": "wide"})
        self.assertIn("obsidian_cssclass: wide", rendered)

    def test_a_value_that_would_break_yaml_is_quoted(self):
        rendered = dump_frontmatter({"summary": "Reads: then writes"})
        self.assertIn('summary: "Reads: then writes"', rendered)


class SectionTests(unittest.TestCase):
    BODY = (
        "# Title\n\n## Quick Summary\n\n- First.\n- Second.\n\n"
        "## Procedure\n\n1. Step one.\n2. Step two.\n\n## Known Problems\n\n- A problem.\n"
    )

    def test_quick_summary_is_found(self):
        self.assertEqual(extract_quick_summary(self.BODY), "- First.\n- Second.")

    def test_a_missing_quick_summary_is_empty_not_guessed(self):
        self.assertEqual(extract_quick_summary("# Title\n\nJust prose.\n"), "")

    def test_one_section_is_read_without_the_next(self):
        self.assertEqual(extract_section(self.BODY, "Procedure"), "1. Step one.\n2. Step two.")

    def test_replacing_a_section_leaves_the_others_intact(self):
        updated = replace_section(self.BODY, "Procedure", "1. A different step.")
        self.assertIn("1. A different step.", updated)
        self.assertNotIn("Step one", updated)
        self.assertIn("- A problem.", updated)
        self.assertIn("- First.", updated)

    def test_replacing_a_missing_section_appends_it(self):
        updated = replace_section(self.BODY, "Lessons Learned", "Something learned.")
        self.assertIn("## Lessons Learned", updated)
        self.assertIn("Something learned.", updated)

    def test_wikilinks_and_list_items(self):
        text = "- [[Python Debugging]]\n- [[Test Verification|tests]]\n- Plain entry\n"
        self.assertEqual(extract_wikilinks(text), ["Python Debugging", "Test Verification"])
        self.assertEqual(extract_list_items(text), ["Python Debugging", "Test Verification", "Plain entry"])


class VaultManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault = VaultManager(Path(self.temp.name) / "vault")
        self.vault.ensure_root()

    def tearDown(self):
        self.temp.cleanup()

    def test_create_read_and_modify(self):
        note = self.vault.create_note(
            "skills/example.md",
            title="Example",
            note_type="skill",
            summary="An example skill note.",
            tags=["example"],
            quick_summary=["Does an example thing."],
            sections=[("Procedure", "1. Example.")],
        )
        self.assertTrue(note.has_summary)
        again = self.vault.read("skills/example.md")
        self.assertEqual(again.title, "Example")
        self.assertEqual(again.section("Procedure"), "1. Example.")

        def mutate(target):
            target.body = replace_section(target.body, "Procedure", "1. Changed.")
            return target

        self.vault.update_note("skills/example.md", mutate)
        self.assertEqual(self.vault.read("skills/example.md").section("Procedure"), "1. Changed.")

    def test_memory_survives_a_new_manager(self):
        """The point of files: a restart loses nothing."""
        self.vault.create_note(
            "user/fact.md", title="Fact", note_type="user", summary="A durable fact.", quick_summary=["Kept."]
        )
        reopened = VaultManager(self.vault.root)
        self.assertEqual(reopened.read("user/fact.md").summary, "A durable fact.")

    def test_a_note_cannot_be_created_without_a_summary(self):
        with self.assertRaises(VaultError):
            self.vault.create_note("skills/bad.md", title="Bad", note_type="skill", summary="   ")

    def test_a_path_outside_the_vault_is_refused(self):
        with self.assertRaises(OutsideVault):
            self.vault.resolve("../../etc/passwd")
        with self.assertRaises(OutsideVault):
            self.vault.note_path("../escape")

    def test_note_exists_is_false_rather_than_raising_for_an_escape(self):
        self.assertFalse(self.vault.note_exists("../../nope"))

    def test_creating_an_existing_note_returns_it_rather_than_overwriting(self):
        first = self.vault.create_note(
            "skills/keep.md", title="Keep", note_type="skill", summary="Original.", quick_summary=["Original."]
        )
        second = self.vault.create_note(
            "skills/keep.md", title="Keep", note_type="skill", summary="Replacement.", quick_summary=["Replacement."]
        )
        self.assertEqual(second.summary, "Original.")
        self.assertEqual(first.summary, second.summary)

    def test_a_malformed_note_does_not_stop_a_scan(self):
        self.vault.create_note("skills/good.md", title="Good", note_type="skill", summary="Fine.", quick_summary=["Fine."])
        (self.vault.root / "skills" / "broken.md").write_text("---\n\tbroken: [\n---\nbody", encoding="utf-8")
        notes = self.vault.notes()
        self.assertEqual(len(notes), 2)
        self.assertTrue(any(note.malformed for note in notes))
        self.assertTrue(any(note.title == "Good" for note in notes))

    def test_moving_a_note_never_clobbers_an_existing_one(self):
        for name in ("a", "b"):
            self.vault.create_note(
                f"missions/active/{name}.md", title=name, note_type="mission", summary=f"Mission {name}.", quick_summary=["x"]
            )
        self.vault.move_note("missions/active/a.md", "missions/completed/a.md")
        self.vault.move_note("missions/active/b.md", "missions/completed/a.md")
        completed = sorted(note.relative_path for note in self.vault.notes("missions/completed"))
        self.assertEqual(len(completed), 2)
        self.assertIn("missions/completed/a.md", completed)
        self.assertIn("missions/completed/a-2.md", completed)

    def test_the_obsidian_config_folder_is_never_scanned(self):
        (self.vault.root / ".obsidian").mkdir()
        (self.vault.root / ".obsidian" / "workspace.md").write_text("not knowledge", encoding="utf-8")
        self.vault.create_note("skills/real.md", title="Real", note_type="skill", summary="Real.", quick_summary=["Real."])
        self.assertEqual([note.title for note in self.vault.notes()], ["Real"])

    def test_a_write_is_atomic(self):
        """A crash mid-write must not leave a truncated note behind.

        Checked by proving the temporary file is gone and the content is
        complete -- the observable consequence of the `os.replace` design.
        """
        self.vault.write_text("state/current.md", "---\ntype: state\n---\n\n# X\n\ncomplete\n")
        leftovers = [item.name for item in (self.vault.root / "state").iterdir() if item.suffix == ".tmp"]
        self.assertEqual(leftovers, [])
        self.assertIn("complete", self.vault.read_text("state/current.md"))


if __name__ == "__main__":
    unittest.main()
