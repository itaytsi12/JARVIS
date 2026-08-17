import tempfile
import unittest
from pathlib import Path

from brain.experience_store import ExperienceRecord, ExperienceStore, retrieve_relevant_experiences
from brain.learning_package import LearningPackage


def _package(**overrides) -> LearningPackage:
    defaults = dict(
        learning_job_id="job-1", improvement_attempt_id="att-1", problem_family="fam-1",
        original_task="fix the browser stale-tab handle crash", subsystem="browser", gap_type="EXECUTION_BUG",
        root_cause_category="EXECUTION_BUG_with_differential_test",
        reusable_strategy="A source_only change touching tools/browser.py resolved an EXECUTION_BUG issue in the browser subsystem.",
        applicability_conditions=["gap_type=EXECUTION_BUG", "subsystem=browser"],
        files_changed=["tools/browser.py"],
    )
    defaults.update(overrides)
    return LearningPackage(**defaults)


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.temp.name) / "exp.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_store_then_get_round_trips(self):
        record = ExperienceRecord.from_package(_package(), experience_id="exp-1")
        self.store.store(record)
        fetched = self.store.get("exp-1")
        self.assertEqual(fetched, record)

    def test_survives_new_connection_same_file(self):
        path = Path(self.temp.name) / "persist.sqlite3"
        store1 = ExperienceStore(path)
        store1.store(ExperienceRecord.from_package(_package(), experience_id="exp-1"))
        store1.close()
        store2 = ExperienceStore(path)
        try:
            self.assertIsNotNone(store2.get("exp-1"))
        finally:
            store2.close()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.temp.name) / "exp.sqlite3")
        self.store.store(ExperienceRecord.from_package(
            _package(original_task="fix the browser stale-tab handle crash", subsystem="browser", gap_type="EXECUTION_BUG"),
            experience_id="browser-fix",
        ))
        self.store.store(ExperienceRecord.from_package(
            _package(original_task="whatsapp message send fails silently", subsystem="messaging", gap_type="EXECUTION_BUG"),
            experience_id="messaging-fix",
        ))
        self.store.store(ExperienceRecord.from_package(
            _package(original_task="calculator division by zero not handled", subsystem="filesystem", gap_type="CODE_CAPABILITY_GAP"),
            experience_id="calc-fix",
        ))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_retrieval_ranks_relevant_experience_first(self):
        results = retrieve_relevant_experiences("browser tab handle is stale after crash", store=self.store, top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].experience.experience_id, "browser-fix")

    def test_subsystem_boost_helps_disambiguate(self):
        results = retrieve_relevant_experiences("something is broken", subsystem="messaging", store=self.store, top_k=1)
        self.assertEqual(results[0].experience.experience_id, "messaging-fix")

    def test_returns_only_top_k_not_the_whole_database(self):
        results = retrieve_relevant_experiences("fix", store=self.store, top_k=1, min_score=0.0)
        self.assertLessEqual(len(results), 1)

    def test_unrelated_query_below_threshold_returns_nothing(self):
        results = retrieve_relevant_experiences("completely unrelated astrophysics question about black holes", store=self.store, top_k=3)
        self.assertEqual(results, [])

    def test_empty_store_returns_empty_list(self):
        empty_store = ExperienceStore(Path(self.temp.name) / "empty.sqlite3")
        try:
            self.assertEqual(retrieve_relevant_experiences("anything", store=empty_store), [])
        finally:
            empty_store.close()


if __name__ == "__main__":
    unittest.main()
