import pytest

from data_sync import DataSyncService, FlakyDataSource
from retry_util import RetryExhaustedError, retry


def test_sync_succeeds_with_flaky_source_hidden():
    service = DataSyncService(FlakyDataSource(fail_times=2))
    assert service.sync() == {"data": "ok"}


def test_sync_succeeds_even_with_no_failures_hidden():
    service = DataSyncService(FlakyDataSource(fail_times=0))
    assert service.sync() == {"data": "ok"}


def test_retry_util_contract_is_unchanged_hidden():
    """Catches a misguided 'fix' that weakens retry_util.retry()'s own
    validation instead of fixing the actual caller bug in data_sync.py."""
    with pytest.raises(ValueError):
        retry(lambda: None, max_attempts=0)

    calls = {"count": 0}

    def always_fails():
        calls["count"] += 1
        raise RuntimeError("permanent failure")

    with pytest.raises(RetryExhaustedError):
        retry(always_fails, max_attempts=3)
    assert calls["count"] == 3
