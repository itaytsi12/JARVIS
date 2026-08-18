from retry_util import retry


class FlakyDataSource:
    """A data source that fails a fixed number of times before recovering
    -- simulates a transient network hiccup, not a permanent failure."""

    def __init__(self, fail_times=2):
        self.fail_times = fail_times
        self.calls = 0

    def fetch(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("temporary failure")
        return {"data": "ok"}


class DataSyncService:
    def __init__(self, source):
        self.source = source

    def sync(self):
        # BUG: misuses retry_util.retry()'s contract. max_attempts=0 means
        # zero attempts, which retry() correctly rejects with ValueError --
        # so sync() always crashes instead of actually retrying the flaky
        # source.
        return retry(self.source.fetch, max_attempts=0)
