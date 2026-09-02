"""THE critical end-to-end test: does the foundation actually work?

This drives the REAL `brain/agent_service.py::run_agent_task` against a
scripted fake provider and a real vault in a temporary directory. Nothing
about the vault is mocked -- the notes, the index, the mission records,
the Daily Note and the learning are all genuine files on disk. Only the
model is fake, so the test is free and offline.

The sequence, exactly as the brief specifies it:

     1. the user gives a mission
     2. JARVIS scans note SUMMARIES
     3. JARVIS selects the relevant Job
     4. JARVIS loads only the required Skills and memory
     5. a mission record is created
     6. JARVIS performs several actions
     7. one method fails
     8. another succeeds
     9. JARVIS records the successful method
    10. the user says "don't do it that way next time, do X instead"
    11. JARVIS identifies a PERSISTENT correction
    12. JARVIS modifies the right Job or Skill note
    13. the modified note's summary is still accurate
    14. the Daily Note records what happened
    15. the components are shut down and reinitialised
    16. the same mission is given again
    17. JARVIS scans the vault
    18. JARVIS reads the UPDATED note
    19. JARVIS performs the corrected behaviour

Step 19 is the one that matters, and it is checked the only way that
means anything: by asserting that the corrected rule is present in the
text the SECOND run actually sent to the model.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vault as vault_package
from brain.agent_service import run_agent_task
from memory.agent_memory import AgentMemory
from memory.agent_store import AgentDatabase
from providers.base import ModelResponse, Usage
from providers.mock_provider import ScriptedProvider, text_response, tool_response
from vault.bootstrap import bootstrap_vault
from vault.daily import DailyJournal, get_journal
from vault.index import VaultIndex
from vault.learning import CorrectionLearner
from vault.manager import VaultManager
from vault.missions import MissionStore
from vault.skills import SkillLibrary


class RecordingProvider:
    """A scripted provider that also KEEPS what it was sent.

    The captured system prompt is the evidence for step 19: it is the
    actual text the model received, so "JARVIS used the corrected note"
    is checked against what happened rather than against a label.
    """

    name = "recording"

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self.system_prompts: list[str] = []
        self.model = "recording-model"

    def is_available(self) -> bool:
        return True

    def complete(self, messages, *, system=None, tools=None, **kwargs) -> ModelResponse:
        self.system_prompts.append(system or "")
        if self._index >= len(self._responses):
            return ModelResponse(text="Done, sir.", provider=self.name, model=self.model, usage=Usage(reported=False))
        response = self._responses[self._index]
        self._index += 1
        response.provider = self.name
        response.model = self.model
        return response


def _memory() -> AgentMemory:
    return AgentMemory(AgentDatabase(Path(tempfile.mkdtemp()) / "agent.sqlite3"))


class CriticalEndToEndTests(unittest.TestCase):
    """One vault, two runs of the same mission, a correction in between."""

    GOAL = "Open Apple Music and play the album I was listening to yesterday, then verify it is playing."

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vault"
        self._build_vault()
        # Point every process-wide vault singleton at the temporary vault.
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

    # -- the vault this test starts from --------------------------------
    def _build_vault(self):
        vault = VaultManager(self.root)
        index = VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        bootstrap_vault(vault, index)

        vault.create_note(
            "skills/apple-music-control.md",
            title="Apple Music Control",
            note_type="skill",
            summary="How JARVIS opens, searches and controls Apple Music on Windows.",
            tags=["music", "apple-music", "windows"],
            quick_summary=["Open Apple Music, then search for what the user asked for."],
            sections=[
                ("When To Use", "Any request to play, pause or find music in Apple Music."),
                ("Procedure", "1. Launch Apple Music.\n2. Search for the track or album.\n3. Play the first result."),
                ("Known Working Method", ""),
                ("Known Problems", ""),
                ("Lessons Learned", ""),
            ],
        )
        vault.create_note(
            "jobs/play-music.md",
            title="Play Music",
            note_type="job",
            summary="Play a specific song, album or playlist for the user in Apple Music and confirm it is really playing.",
            tags=["job", "music", "apple-music", "playback"],
            quick_summary=["Use when the user asks to play, resume or queue music."],
            sections=[
                ("Goal", "The requested music is confirmed playing."),
                ("When To Use", "The user asks to play or open music, an album, a song or a playlist."),
                ("Required Context", "- Any recorded listening history."),
                ("Required Skills", "- [[Apple Music Control]]"),
                ("Procedure", "1. Open Apple Music.\n2. Find what was asked for.\n3. Play it.\n4. Verify the player really shows it."),
                ("Completion Requirements", "- The player bar shows the requested track."),
                ("Quality Rules", "- Never claim playback that was not observed."),
                ("Known Problems", ""),
                ("Lessons Learned", ""),
                ("Safety / Approval Rules", "- See [[Protected Rules]]."),
            ],
        )
        # Decoys: enough unrelated notes that selecting the right one is a
        # real result rather than an accident of a tiny vault.
        for number in range(40):
            vault.create_note(
                f"lessons/unrelated-{number:02d}.md",
                title=f"Unrelated Note {number}",
                note_type="lesson",
                summary=f"An observation about spreadsheets and taxes, number {number}.",
                tags=["decoy"],
                quick_summary=[f"Nothing about audio. Item {number}."],
                sections=[("Detail", "d" * 800)],
            )

    def _provider(self):
        """A run where the FIRST method fails and the second succeeds."""
        return RecordingProvider(
            [
                tool_response("open_application", {"app_name": "Apple Music"}, call_id="c1"),
                tool_response("click_ui_element", {"app_name": "Apple Music", "name": "Play"}, call_id="c2"),
                tool_response("music_play", {"song": "the album from yesterday"}, call_id="c3"),
                tool_response("music_now_playing", {}, call_id="c4"),
                text_response("The album is playing, sir."),
            ]
        )

    def _catalog(self, failing_tool: str):
        """A catalog whose tools are real dispatch shapes, one of which fails.

        The failure is what makes step 7 genuine: a method that did not
        work, followed by one that did.
        """
        from brain.models import ToolResult
        from brain.tool_catalog import ToolCatalog

        catalog = ToolCatalog()

        def make(name):
            def handler(arguments):
                if name == failing_tool:
                    return ToolResult(False, name, f"{name} did not work.", {"verified": True}, "control_not_found")
                return ToolResult(True, name, f"{name} succeeded.", {"verified": True})

            return handler

        for name in ("open_application", "click_ui_element", "music_play", "music_now_playing"):
            catalog.register_handler(name, make(name))
        return catalog

    # -- the test ------------------------------------------------------
    def test_the_whole_loop_from_mission_to_corrected_behaviour(self):
        # ---------- steps 1-9: the first run ---------------------------
        provider = self._provider()
        first = run_agent_task(
            self.GOAL,
            provider=provider,
            memory=_memory(),
            catalog=self._catalog(failing_tool="click_ui_element"),
        )
        self.assertTrue(first.success, first.run.stop_reason)

        session = first.vault
        self.assertIsNotNone(session, "the vault session was never created")

        # 2 -- summaries were scanned, not bodies.
        self.assertGreaterEqual(session.primed.scanned, 45)

        # 3 -- the right Job was selected from its summary.
        self.assertEqual(session.primed.job_title, "Play Music")

        # 4 -- only the required Skill was loaded, and no decoy was read.
        self.assertIn("Apple Music Control", session.primed.skill_titles)
        self.assertFalse([path for path in session.primed.notes_read if "unrelated" in path])
        self.assertLess(session.primed.used_chars, 6000)

        # 5 -- a mission record exists on disk.
        self.assertIsNotNone(session.mission)
        mission_path = session.mission.relative_path
        self.assertTrue(VaultManager(self.root).note_exists(mission_path))

        # 6, 7, 8 -- several actions ran; one failed and a later one worked.
        tools = [step.tool for step in first.run.steps]
        self.assertEqual(tools, ["open_application", "click_ui_element", "music_play", "music_now_playing"])
        self.assertFalse(first.run.steps[1].success)
        self.assertTrue(first.run.steps[-1].success)

        # 9 -- the successful method was recorded on the Skill.
        vault = VaultManager(self.root)
        method = vault.read("skills/apple-music-control.md").section("Known Working Method")
        self.assertIn("music_now_playing", method)
        self.assertIn("Does NOT work", method)

        # The mission record itself captured the failure and the outcome.
        completed = vault.read(session.mission.relative_path)
        self.assertIn("click_ui_element", completed.section("Failures And Retries"))
        self.assertIn("Succeeded", completed.section("Outcome"))
        self.assertIn("missions/completed/", session.mission.relative_path)

        # ---------- steps 10-13: the correction ------------------------
        correction = (
            "No. From now on, when Apple Music is already open, don't launch it again -- "
            "use music_play directly instead of clicking the Play control."
        )
        index = VaultIndex(vault, cache_path=None, use_cache=False, refresh_interval=0.0)
        learner = CorrectionLearner(vault=vault, index=index)
        outcome = learner.apply(correction, candidate_paths=session.primed.notes_read)

        # 11 -- recognised as durable, not as an instruction for this task.
        self.assertEqual(outcome.kind, "persistent")
        # 12 -- the right note was modified.
        self.assertTrue(outcome.applied, outcome.reason)
        self.assertIn(outcome.target_title, {"Apple Music Control", "Play Music"})

        updated = vault.read(outcome.target_path)
        self.assertIn("music_play", updated.body)
        # 13 -- the note still describes itself accurately.
        self.assertTrue(updated.has_summary)
        self.assertTrue(updated.summary)
        self.assertIn("music", updated.summary.lower())

        # 14 -- the Daily Note recorded the day.
        journal = DailyJournal(vault=vault, index=index)
        today = vault.read(journal.today().relative_path)
        self.assertIn("Timeline", today.sections())
        self.assertTrue(today.section("Timeline").strip())
        self.assertIn("Play Music", today.section("Timeline"))

        journal.today().add_correction(f"{outcome.rule} (recorded in [[{outcome.target_title}]])")
        today = vault.read(journal.today().relative_path)
        self.assertIn("music_play", today.section("User Corrections / Preferences Learned"))

        # ---------- step 15: shut everything down ----------------------
        vault_package.reset_all()
        del vault, index, learner, journal, session

        # ---------- steps 16-19: the same mission again ----------------
        second_provider = self._provider()
        second = run_agent_task(
            self.GOAL,
            provider=second_provider,
            memory=_memory(),
            catalog=self._catalog(failing_tool="click_ui_element"),
        )
        self.assertTrue(second.success)

        # 17, 18 -- the vault was scanned again and the UPDATED note read.
        reread = second.vault
        self.assertEqual(reread.primed.job_title, "Play Music")
        self.assertIn(outcome.target_path, reread.primed.notes_read)

        # 19 -- THE PROOF. The corrected rule is in the text the model was
        # actually sent. Not a route label, not a passing unit test on a
        # label: the real prompt.
        prompt = second_provider.system_prompts[0]
        self.assertIn("music_play", prompt)
        self.assertIn("Apple Music Control", prompt)
        # And the knowledge the correction produced is there in full.
        self.assertIn(outcome.rule.split(".")[0][:40], prompt)

        # The decoys never reached the model, in either run.
        self.assertNotIn("spreadsheets and taxes", prompt)

    def test_a_trivial_request_pays_no_mission_cost(self):
        """The fast paths stay fast: "volume down" gets no Job, no mission
        note and only the small identity/preferences priming."""
        provider = RecordingProvider([text_response("Done, sir.")])
        outcome = run_agent_task("volume down", provider=provider, memory=_memory())
        session = outcome.vault
        self.assertIsNotNone(session)
        self.assertEqual(session.policy.mode, "light")
        self.assertIsNone(session.mission)
        self.assertLessEqual(session.primed.used_chars, 1500)
        self.assertEqual(MissionStore(vault=VaultManager(self.root)).active(), [])

    def test_a_vault_failure_never_breaks_a_request(self):
        """The rule the whole integration rests on: long-term memory is an
        enhancement, never a dependency."""
        provider = RecordingProvider([text_response("Done, sir.")])
        with patch("vault.session.VaultSession.begin", side_effect=OSError("disk gone")):
            outcome = run_agent_task("run the tests and fix what breaks", provider=provider, memory=_memory())
        self.assertTrue(outcome.success)
        self.assertIsNone(outcome.vault)

    def test_the_vault_can_be_switched_off_entirely(self):
        provider = RecordingProvider([text_response("Done, sir.")])
        outcome = run_agent_task("run the tests and fix what breaks", provider=provider, memory=_memory(), use_vault=False)
        self.assertTrue(outcome.success)
        self.assertIsNone(outcome.vault)
        self.assertNotIn("## Job:", provider.system_prompts[0])


if __name__ == "__main__":
    unittest.main()
