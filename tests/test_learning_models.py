import tempfile
import unittest
from pathlib import Path

from brain.learning_models import (
    ApprovalStatus, LearningJob, LearningJobStatus,
    TERMINAL_LEARNING_JOB_STATUSES, TRAINABLE_LEARNING_JOB_STATUSES,
)
from brain.learning_store import LearningJobStore


def _job(**overrides) -> LearningJob:
    defaults = dict(
        learning_job_id="job-1", created_at="t", updated_at="t",
        candidate_id="cand-1", improvement_attempt_id="att-1",
        fingerprint="fp-1",
    )
    defaults.update(overrides)
    return LearningJob(**defaults)


class LearningJobModelTests(unittest.TestCase):
    def test_round_trips_through_dict(self):
        job = _job(original_request="fix the thing", claude_teacher_used=True)
        restored = LearningJob.from_dict(job.to_dict())
        self.assertEqual(restored, job)

    def test_default_status_is_pending_approval(self):
        job = _job()
        self.assertEqual(job.learning_status, LearningJobStatus.PENDING_APPROVAL.value)
        self.assertEqual(job.approval_status, ApprovalStatus.PENDING.value)

    def test_unknown_fields_in_payload_are_ignored_on_load(self):
        payload = _job().to_dict()
        payload["some_future_field"] = "value"
        restored = LearningJob.from_dict(payload)
        self.assertEqual(restored.learning_job_id, "job-1")

    def test_terminal_and_trainable_status_sets_are_disjoint(self):
        self.assertEqual(TERMINAL_LEARNING_JOB_STATUSES & TRAINABLE_LEARNING_JOB_STATUSES, set())


class LearningJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = LearningJobStore(Path(self.temp.name) / "jobs.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_create_then_get_round_trips(self):
        job = _job()
        self.store.create(job)
        fetched = self.store.get("job-1")
        self.assertEqual(fetched, job)

    def test_create_twice_raises(self):
        self.store.create(_job())
        with self.assertRaises(Exception):
            self.store.create(_job())

    def test_update_persists_status_change(self):
        job = _job()
        self.store.create(job)
        job.learning_status = LearningJobStatus.APPROVED.value
        job.approval_status = ApprovalStatus.APPROVED.value
        self.store.update(job)
        fetched = self.store.get("job-1")
        self.assertEqual(fetched.learning_status, LearningJobStatus.APPROVED.value)

    def test_update_unknown_job_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.store.update(_job(learning_job_id="does-not-exist"))

    def test_survives_new_connection_same_file(self):
        path = Path(self.temp.name) / "persist.sqlite3"
        store1 = LearningJobStore(path)
        store1.create(_job())
        store1.close()
        store2 = LearningJobStore(path)
        try:
            self.assertIsNotNone(store2.get("job-1"))
        finally:
            store2.close()

    def test_find_active_by_fingerprint_ignores_declined_and_timed_out(self):
        self.store.create(_job(learning_job_id="a", fingerprint="fp-x", learning_status=LearningJobStatus.DECLINED.value))
        self.store.create(_job(learning_job_id="b", fingerprint="fp-x", learning_status=LearningJobStatus.APPROVAL_TIMED_OUT.value))
        self.assertIsNone(self.store.find_active_by_fingerprint("fp-x"))

    def test_find_active_by_fingerprint_finds_approved(self):
        self.store.create(_job(learning_job_id="a", fingerprint="fp-y", learning_status=LearningJobStatus.APPROVED.value))
        found = self.store.find_active_by_fingerprint("fp-y")
        self.assertIsNotNone(found)
        self.assertEqual(found.learning_job_id, "a")

    def test_find_active_by_fingerprint_finds_trained(self):
        self.store.create(_job(learning_job_id="a", fingerprint="fp-z", learning_status=LearningJobStatus.TRAINED.value))
        found = self.store.find_active_by_fingerprint("fp-z")
        self.assertIsNotNone(found)

    def test_query_trainable_only_returns_approved_and_ready_statuses(self):
        self.store.create(_job(learning_job_id="a", fingerprint="fp-1", learning_status=LearningJobStatus.APPROVED.value))
        self.store.create(_job(learning_job_id="b", fingerprint="fp-2", learning_status=LearningJobStatus.READY_FOR_TRAINING.value))
        self.store.create(_job(learning_job_id="c", fingerprint="fp-3", learning_status=LearningJobStatus.TRAINED.value))
        self.store.create(_job(learning_job_id="d", fingerprint="fp-4", learning_status=LearningJobStatus.PENDING_APPROVAL.value))
        trainable = {j.learning_job_id for j in self.store.query_trainable()}
        self.assertEqual(trainable, {"a", "b"})

    def test_query_by_status(self):
        self.store.create(_job(learning_job_id="a", learning_status=LearningJobStatus.APPROVED.value))
        self.store.create(_job(learning_job_id="b", fingerprint="fp-2", learning_status=LearningJobStatus.DECLINED.value))
        approved = self.store.query(learning_status=LearningJobStatus.APPROVED.value)
        self.assertEqual([j.learning_job_id for j in approved], ["a"])

    def test_upsert_creates_then_updates(self):
        job = _job()
        self.store.upsert(job)
        job.learning_status = LearningJobStatus.APPROVED.value
        self.store.upsert(job)
        self.assertEqual(self.store.get("job-1").learning_status, LearningJobStatus.APPROVED.value)
        self.assertEqual(self.store.count(), 1)


if __name__ == "__main__":
    unittest.main()
