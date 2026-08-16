import tempfile
import unittest
from pathlib import Path

from brain.improvement_attempt_models import AttemptStatus, ImprovementAttempt
from brain.improvement_attempt_store import ImprovementAttemptStore, reset_improvement_attempt_store_for_tests


def _attempt(**overrides) -> ImprovementAttempt:
    defaults = dict(attempt_id="a1", candidate_id="c1", created_at="2026-08-17T00:00:00+00:00")
    defaults.update(overrides)
    return ImprovementAttempt(**defaults)


class ImprovementAttemptStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ImprovementAttemptStore(Path(self.temp.name) / "attempts.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_create_then_get_round_trips(self):
        self.store.create(_attempt())
        fetched = self.store.get("a1")
        self.assertEqual(fetched.candidate_id, "c1")
        self.assertEqual(fetched.status, AttemptStatus.QUEUED.value)

    def test_get_missing_attempt_returns_none(self):
        self.assertIsNone(self.store.get("does-not-exist"))

    def test_create_duplicate_attempt_id_raises(self):
        self.store.create(_attempt())
        with self.assertRaises(Exception):
            self.store.create(_attempt())

    def test_update_persists_new_status_and_fields(self):
        self.store.create(_attempt())
        updated = _attempt(status=AttemptStatus.READY_FOR_REVIEW.value, files_changed=["brain/agent.py"])
        self.store.update(updated)
        fetched = self.store.get("a1")
        self.assertEqual(fetched.status, AttemptStatus.READY_FOR_REVIEW.value)
        self.assertEqual(fetched.files_changed, ["brain/agent.py"])

    def test_update_missing_attempt_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.store.update(_attempt(attempt_id="never-created"))

    def test_update_never_touches_a_different_attempt(self):
        self.store.create(_attempt(attempt_id="a1"))
        self.store.create(_attempt(attempt_id="a2"))
        self.store.update(_attempt(attempt_id="a1", status=AttemptStatus.CANCELLED.value))
        self.assertEqual(self.store.get("a2").status, AttemptStatus.QUEUED.value)

    def test_list_for_candidate_returns_only_matching_attempts_newest_first(self):
        self.store.create(_attempt(attempt_id="a1", candidate_id="shared", created_at="2026-01-01T00:00:00+00:00"))
        self.store.create(_attempt(attempt_id="a2", candidate_id="shared", created_at="2026-06-01T00:00:00+00:00"))
        self.store.create(_attempt(attempt_id="a3", candidate_id="other", created_at="2026-01-01T00:00:00+00:00"))
        attempts = self.store.list_for_candidate("shared")
        self.assertEqual([a.attempt_id for a in attempts], ["a2", "a1"])

    def test_has_active_attempt_true_for_non_terminal_status(self):
        self.store.create(_attempt(status=AttemptStatus.IMPROVING.value))
        self.assertTrue(self.store.has_active_attempt("c1"))

    def test_has_active_attempt_false_once_terminal(self):
        self.store.create(_attempt(status=AttemptStatus.FIX_FAILED.value))
        self.assertFalse(self.store.has_active_attempt("c1"))

    def test_has_active_attempt_false_for_unrelated_candidate(self):
        self.store.create(_attempt(candidate_id="c1", status=AttemptStatus.IMPROVING.value))
        self.assertFalse(self.store.has_active_attempt("c2"))

    def test_has_ready_for_review_attempt(self):
        self.store.create(_attempt(status=AttemptStatus.READY_FOR_REVIEW.value))
        self.assertTrue(self.store.has_ready_for_review_attempt("c1"))
        self.assertFalse(self.store.has_active_attempt("c1"))  # terminal, not "active"

    def test_query_filters_by_status(self):
        self.store.create(_attempt(attempt_id="a1", status=AttemptStatus.QUEUED.value))
        self.store.create(_attempt(attempt_id="a2", status=AttemptStatus.FIX_FAILED.value))
        queued = self.store.query(status=AttemptStatus.QUEUED.value)
        self.assertEqual([a.attempt_id for a in queued], ["a1"])

    def test_count(self):
        self.assertEqual(self.store.count(), 0)
        self.store.create(_attempt(attempt_id="a1"))
        self.store.create(_attempt(attempt_id="a2", candidate_id="c2"))
        self.assertEqual(self.store.count(), 2)

    def test_persists_across_new_connection_same_file(self):
        path = Path(self.temp.name) / "reopened.sqlite3"
        store_one = ImprovementAttemptStore(path)
        store_one.create(_attempt())
        store_one.close()
        store_two = ImprovementAttemptStore(path)
        self.assertEqual(store_two.get("a1").candidate_id, "c1")
        store_two.close()


class ImprovementAttemptStoreSingletonTests(unittest.TestCase):
    def test_reset_for_tests_gives_isolated_store(self):
        # mkdtemp (not TemporaryDirectory) deliberately: the singleton this
        # helper installs outlives this test, so nothing here should try to
        # auto-delete the directory while that connection is still open.
        tmp = Path(tempfile.mkdtemp(prefix="jarvis-improvement-attempt-store-test-"))
        store = reset_improvement_attempt_store_for_tests(tmp / "x.sqlite3")
        store.create(_attempt())
        self.assertEqual(store.count(), 1)


if __name__ == "__main__":
    unittest.main()
