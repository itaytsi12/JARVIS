"""Memory: extraction rules, persistence across restart, and retrieval."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory.agent_memory import AgentMemory
from memory.agent_store import AgentDatabase
from memory.conversation import ASSISTANT, USER, ConversationStore
from memory.episodic import Episode, EpisodeStore, StepRecord
from memory.long_term import (
    CORRECTION,
    EXPLICIT,
    PREFERENCE,
    PROJECT,
    LongTermMemoryStore,
    extract_memories,
)
from memory.retrieval import rank_episodes, rank_memories, score_text


def _database() -> AgentDatabase:
    return AgentDatabase(Path(tempfile.mkdtemp()) / "agent.sqlite3")


class ExtractionTests(unittest.TestCase):
    def test_ordinary_commands_are_never_promoted_to_memory(self):
        for command in (
            "open YouTube",
            "volume down",
            "play Starboy",
            "take a screenshot",
            "close notepad",
            "calculate 527 * 93",
            "what time is it",
        ):
            with self.subTest(command=command):
                self.assertEqual(extract_memories(command), [])

    def test_explicit_instructions_are_always_kept(self):
        found = extract_memories("Remember that this is my main Jarvis project")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].source, EXPLICIT)
        self.assertEqual(found[0].kind, PROJECT)
        self.assertEqual(found[0].text, "this is my main Jarvis project")
        self.assertGreaterEqual(found[0].importance, 5)

    def test_from_now_on_is_an_explicit_instruction(self):
        found = extract_memories("From now on keep spoken answers under two sentences")
        self.assertEqual(found[0].source, EXPLICIT)

    def test_preferences_are_kept(self):
        found = extract_memories("I prefer short spoken answers")
        self.assertEqual(found[0].kind, PREFERENCE)

    def test_corrections_are_kept(self):
        found = extract_memories("No, I meant the other project")
        self.assertEqual(found[0].kind, CORRECTION)

    def test_stated_facts_are_kept(self):
        found = extract_memories("My main repository is at C:/dev/jarvis")
        self.assertTrue(found)
        self.assertIn(found[0].kind, {"fact", PROJECT})

    def test_a_command_phrased_like_a_statement_is_still_not_remembered(self):
        self.assertEqual(extract_memories("play my gym playlist"), [])

    def test_empty_input_is_safe(self):
        self.assertEqual(extract_memories(""), [])
        self.assertEqual(extract_memories("ok"), [])


class LongTermStoreTests(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        self.store = LongTermMemoryStore(self.database)

    def test_a_memory_round_trips(self):
        record = self.store.remember("my project is at C:/dev/jarvis", kind=PROJECT, importance=4)
        self.assertEqual(self.store.get(record.memory_id).text, "my project is at C:/dev/jarvis")

    def test_repeating_a_fact_strengthens_it_instead_of_duplicating(self):
        self.store.remember("I use vscode", kind=PREFERENCE, importance=2)
        self.store.remember("I use vscode", kind=PREFERENCE, importance=5)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.by_kind(PREFERENCE)[0].importance, 5)

    def test_secrets_are_redacted_before_storage(self):
        record = self.store.remember("my api key is sk-abcdefghijklmnop1234")
        self.assertNotIn("sk-abcdefghijklmnop1234", record.text)

    def test_forgetting_retires_rather_than_deletes(self):
        record = self.store.remember("temporary belief")
        self.store.forget(record.memory_id)
        self.assertEqual(self.store.count(), 0)
        self.assertIsNotNone(self.store.get(record.memory_id))

    def test_memory_survives_a_reopen(self):
        self.store.remember("persisted fact")
        reopened = LongTermMemoryStore(AgentDatabase(self.database.path))
        self.assertEqual(reopened.count(), 1)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        self.store = ConversationStore(self.database)

    def test_turns_are_ordered_and_persisted(self):
        session = self.store.start_session()
        self.store.add_turn(session, USER, "open youtube")
        self.store.add_turn(session, ASSISTANT, "Opening YouTube.")
        turns = self.store.recent_turns(session, 10)
        self.assertEqual([turn.role for turn in turns], [USER, ASSISTANT])
        self.assertEqual([turn.sequence for turn in turns], [1, 2])

    def test_recent_turns_are_bounded_and_newest_last(self):
        session = self.store.start_session()
        for index in range(10):
            self.store.add_turn(session, USER, f"message {index}")
        turns = self.store.recent_turns(session, 3)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[-1].text, "message 9")

    def test_history_survives_a_restart(self):
        session = self.store.start_session()
        self.store.add_turn(session, USER, "remember the plan")
        reopened = ConversationStore(AgentDatabase(self.database.path))
        self.assertEqual(reopened.turn_count(session), 1)

    def test_search_finds_an_earlier_turn(self):
        session = self.store.start_session()
        self.store.add_turn(session, USER, "the voice bug is in elevenlabs auth")
        self.assertTrue(self.store.search("elevenlabs"))

    def test_sessions_are_listed(self):
        session = self.store.start_session(title="first")
        self.store.end_session(session)
        listed = self.store.sessions()
        self.assertEqual(listed[0]["session_id"], session)
        self.assertIsNotNone(listed[0]["ended_at"])


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        self.jsonl = Path(tempfile.mkdtemp()) / "episodes.jsonl"
        self.store = EpisodeStore(self.database, self.jsonl)

    def _episode(self, **overrides) -> Episode:
        payload = dict(
            episode_id="",
            user_request="fix the failing test",
            route="agent",
            model_used="claude-opus-5",
            success=True,
            verified=True,
            duration_ms=1234.5,
            final_result="The test passes now.",
            steps=[StepRecord(0, "run_command", {"command": "pytest"}, True, True, "1 passed")],
            token_usage={"input_tokens": 100, "output_tokens": 20},
            estimated_cost_usd=0.001,
        )
        payload.update(overrides)
        return Episode(**payload)

    def test_an_episode_round_trips_with_every_field(self):
        stored = self.store.record(self._episode())
        loaded = self.store.get(stored.episode_id)
        self.assertEqual(loaded.user_request, "fix the failing test")
        self.assertEqual(loaded.steps[0].tool, "run_command")
        self.assertEqual(loaded.token_usage["input_tokens"], 100)
        self.assertEqual(loaded.model_used, "claude-opus-5")

    def test_raw_jsonl_is_written_for_future_training(self):
        self.store.record(self._episode())
        lines = self.jsonl.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        import json

        payload = json.loads(lines[0])
        for key in ("user_request", "steps", "success", "duration_ms", "token_usage", "estimated_cost_usd"):
            self.assertIn(key, payload)

    def test_reward_relevant_signals_are_derived(self):
        episode = self._episode(
            success=False,
            retries=2,
            errors=["run_command:exit_code_1"],
            steps=[StepRecord(0, "run_command", {}, False, True, "failed", error="exit_code_1")],
        )
        self.assertEqual(episode.error_count, 2)
        self.assertEqual(episode.step_count, 1)
        stored = self.store.record(episode)
        self.assertFalse(self.store.get(stored.episode_id).success)

    def test_statistics_summarize_outcomes(self):
        self.store.record(self._episode())
        self.store.record(self._episode(success=False, verified=False))
        statistics = self.store.statistics()
        self.assertEqual(statistics["episodes"], 2)
        self.assertEqual(statistics["successes"], 1)
        self.assertAlmostEqual(statistics["success_rate"], 0.5)

    def test_episodes_survive_a_restart(self):
        self.store.record(self._episode())
        reopened = EpisodeStore(AgentDatabase(self.database.path), self.jsonl)
        self.assertEqual(reopened.count(), 1)


class RetrievalTests(unittest.TestCase):
    def test_score_rewards_shared_distinctive_tokens(self):
        self.assertGreater(
            score_text("fix the elevenlabs websocket bug", "the elevenlabs websocket auth race"),
            score_text("fix the elevenlabs websocket bug", "play some music"),
        )

    def test_stopwords_alone_do_not_create_relevance(self):
        self.assertEqual(score_text("the and of", "the and of"), 0.0)

    def test_relevant_memories_outrank_irrelevant_ones(self):
        store = LongTermMemoryStore(_database())
        store.remember("my jarvis project lives at C:/dev/jarvis", kind=PROJECT, importance=4)
        store.remember("I like my coffee black", importance=2)
        ranked = rank_memories("open my jarvis project", store.all(), top_k=1)
        self.assertIn("jarvis", ranked[0].memory.text)

    def test_an_episode_with_no_overlap_is_not_returned(self):
        recent = Episode(episode_id="e1", user_request="fix the voice bug", success=True, created_at=_now())
        self.assertEqual(rank_episodes("what is the weather", [recent]), [])

    def test_a_relevant_episode_is_returned(self):
        recent = Episode(episode_id="e1", user_request="fix the voice bug", success=True, created_at=_now())
        ranked = rank_episodes("continue fixing the voice bug", [recent])
        self.assertEqual(ranked[0].episode.episode_id, "e1")

    def test_recency_breaks_ties_between_equally_relevant_episodes(self):
        old = Episode(episode_id="old", user_request="fix the voice bug", success=True, created_at=_now(days=30))
        new = Episode(episode_id="new", user_request="fix the voice bug", success=True, created_at=_now())
        ranked = rank_episodes("fix the voice bug", [old, new], top_k=2)
        self.assertEqual(ranked[0].episode.episode_id, "new")


class AgentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.database = _database()
        self.memory = AgentMemory(self.database)

    def test_an_ordinary_command_is_stored_as_history_but_not_as_memory(self):
        written = self.memory.observe_exchange("open youtube", "Opening YouTube.")
        self.assertEqual(written, [])
        self.assertEqual(self.memory.long_term.count(), 0)
        self.assertEqual(self.memory.conversation.turn_count(self.memory.session_id), 2)

    def test_an_explicit_instruction_becomes_a_durable_memory(self):
        written = self.memory.observe_exchange("Remember that my main project is at C:/dev/jarvis", "Noted.")
        self.assertEqual(len(written), 1)
        self.assertEqual(self.memory.long_term.count(), 1)

    def test_retrieval_returns_a_bounded_relevant_slice(self):
        self.memory.observe_exchange("Remember that my main project is at C:/dev/jarvis")
        self.memory.record_episode(
            Episode(episode_id="", user_request="fix the voice bug", success=True, final_result="fixed the auth race")
        )
        retrieved = self.memory.retrieve("continue fixing the voice bug", max_memories=2, max_episodes=2)
        self.assertTrue(retrieved.episodes)
        self.assertLessEqual(len(retrieved.memories), 2)
        self.assertIn("episode_ids", retrieved.describe())

    def test_everything_survives_a_process_restart(self):
        self.memory.observe_exchange("Remember that I use vscode", "Noted.")
        self.memory.record_episode(Episode(episode_id="", user_request="ran the tests", success=True))
        restarted = AgentMemory(AgentDatabase(self.database.path))
        self.assertEqual(restarted.long_term.count(), 1)
        self.assertEqual(restarted.episodes.count(), 1)
        self.assertGreater(restarted.conversation.turn_count(), 0)

    def test_statistics_report_the_whole_memory_system(self):
        statistics = self.memory.statistics()
        for key in ("long_term_memories", "conversation_turns", "episodes"):
            self.assertIn(key, statistics)


def _now(days: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


if __name__ == "__main__":
    unittest.main()
