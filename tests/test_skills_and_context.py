"""Skills selection and context budgeting."""
import unittest

from brain.context_builder import BASE_SYSTEM_PROMPT, ContextBuilder
from brain.tool_catalog import ToolCatalog
from memory.agent_memory import RetrievedContext
from memory.conversation import ConversationTurn
from memory.episodic import Episode
from memory.long_term import EXPLICIT, MemoryRecord
from memory.retrieval import ScoredEpisode, ScoredMemory
from skills import get_skill_registry
from skills.base import Skill, SkillRegistry


class SkillSelectionTests(unittest.TestCase):
    def setUp(self):
        self.registry = get_skill_registry()

    def test_every_skill_declares_tools_that_exist(self):
        catalog = ToolCatalog()
        for skill in self.registry.all():
            with self.subTest(skill=skill.name):
                self.assertTrue(skill.tools(catalog), f"{skill.name} resolved to no tools")
                self.assertTrue(skill.guidance.strip())

    def test_a_coding_goal_selects_the_coding_skill(self):
        for goal in (
            "run my project and fix the crash",
            "why does this function throw an exception",
            "run pytest and tell me what fails",
            "fix the bug in music_intent.py",
        ):
            with self.subTest(goal=goal):
                self.assertIn("coding", [skill.name for skill in self.registry.select(goal)])

    def test_a_desktop_goal_selects_computer_control(self):
        self.assertIn("computer_control", [skill.name for skill in self.registry.select("open spotify and turn the volume up")])

    def test_a_file_goal_selects_the_files_skill(self):
        self.assertIn("files", [skill.name for skill in self.registry.select("organize the files in my downloads folder")])

    def test_an_unrelated_goal_selects_nothing_rather_than_guessing(self):
        self.assertEqual(self.registry.select("qwerty zxcvbn plugh"), [])

    def test_selection_is_bounded(self):
        self.assertLessEqual(len(self.registry.select("fix the code and open the browser and organize files", limit=2)), 2)

    def test_a_new_skill_can_be_registered_without_touching_the_runtime(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                name="gardening",
                description="water the plants",
                guidance="Water them.",
                tool_names=("get_time",),
                keywords=("water", "plants"),
            )
        )
        self.assertEqual([skill.name for skill in registry.select("water the plants")], ["gardening"])
        self.assertEqual([definition.name for definition in registry.get("gardening").tools()], ["get_time"])

    def test_catalog_summary_lists_every_skill(self):
        summary = self.registry.catalog_summary()
        for name in self.registry.names():
            self.assertIn(name, summary)


class ContextBuildingTests(unittest.TestCase):
    def _retrieved(self, memories=2, episodes=1, turns=2) -> RetrievedContext:
        return RetrievedContext(
            query="fix the voice bug",
            memories=[
                ScoredMemory(
                    MemoryRecord(f"m{index}", "project", f"memory number {index}", source=EXPLICIT), 0.9 - index * 0.1
                )
                for index in range(memories)
            ],
            episodes=[
                ScoredEpisode(
                    Episode(episode_id=f"e{index}", user_request="fix the voice bug", success=True, final_result="fixed it"),
                    0.8,
                )
                for index in range(episodes)
            ],
            recent_turns=[
                ConversationTurn(f"t{index}", "s", index, "user", f"turn {index}", "2026-01-01T00:00:00+00:00")
                for index in range(turns)
            ],
        )

    def test_the_base_prompt_and_request_are_always_present(self):
        built = ContextBuilder().build("open spotify")
        self.assertIn(BASE_SYSTEM_PROMPT, built.system_prompt)
        self.assertEqual(built.user_prompt, "open spotify")

    def test_retrieved_memory_episodes_and_turns_are_included(self):
        built = ContextBuilder().build("fix the voice bug", retrieved=self._retrieved())
        self.assertIn("What you know about this user", built.system_prompt)
        self.assertIn("Related things you did before", built.system_prompt)
        self.assertIn("Recent conversation", built.system_prompt)

    def test_skill_guidance_is_included_with_completion_criteria(self):
        skill = get_skill_registry().get("coding")
        built = ContextBuilder().build("fix the bug", skills=[skill])
        self.assertIn("Skill: coding", built.system_prompt)
        self.assertIn("What counts as done", built.system_prompt)
        self.assertEqual(built.skills, ["coding"])

    def test_tool_names_are_advertised(self):
        built = ContextBuilder().build("do a thing", tool_names=["run_command", "read_code"])
        self.assertIn("run_command", built.system_prompt)

    def test_the_budget_is_enforced_and_what_was_cut_is_reported(self):
        builder = ContextBuilder(budget_chars=len(BASE_SYSTEM_PROMPT) + 400)
        built = builder.build("fix the voice bug", retrieved=self._retrieved(memories=30, episodes=20, turns=30))
        self.assertLessEqual(built.used_chars, builder.budget_chars + 400)
        self.assertTrue(built.dropped or any(section.truncated for section in built.sections))

    def test_context_never_grows_without_bound(self):
        small = ContextBuilder().build("x", retrieved=self._retrieved(memories=1, episodes=1, turns=1))
        large = ContextBuilder().build("x", retrieved=self._retrieved(memories=200, episodes=200, turns=200))
        self.assertLessEqual(len(large.system_prompt), ContextBuilder().budget_chars + len(BASE_SYSTEM_PROMPT))
        self.assertGreaterEqual(len(large.system_prompt), len(small.system_prompt))

    def test_the_highest_priority_sections_survive_a_tight_budget(self):
        skill = get_skill_registry().get("coding")
        builder = ContextBuilder(budget_chars=len(BASE_SYSTEM_PROMPT) + len(skill.guidance) + 600)
        built = builder.build("fix the bug", retrieved=self._retrieved(memories=50, turns=50), skills=[skill])
        self.assertIn("Skill: coding", built.system_prompt)
        self.assertIn("conversation", built.dropped + [s.name for s in built.sections if s.truncated])

    def test_desktop_state_is_described_when_available(self):
        class Context:
            active_app = "notepad"
            last_opened_app = "notepad"
            current_url = None
            last_opened_file = None

        built = ContextBuilder().build("type hello", session_context=Context())
        self.assertIn("Active application: notepad", built.system_prompt)

    def test_observations_are_bounded_head_and_tail(self):
        builder = ContextBuilder()
        text = "HEAD" + ("x" * 50000) + "TAIL"
        bounded = builder.bound_observation(text)
        self.assertLess(len(bounded), len(text))
        self.assertTrue(bounded.startswith("HEAD"))
        self.assertTrue(bounded.endswith("TAIL"))
        self.assertIn("characters omitted", bounded)

    def test_describe_is_log_safe_and_informative(self):
        described = ContextBuilder().build("x", retrieved=self._retrieved()).describe()
        for key in ("budget_chars", "used_chars", "sections", "dropped", "skills", "tool_count"):
            self.assertIn(key, described)


if __name__ == "__main__":
    unittest.main()
