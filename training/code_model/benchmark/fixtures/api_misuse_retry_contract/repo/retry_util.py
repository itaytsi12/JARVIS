class RetryExhaustedError(Exception):
    pass


def retry(fn, max_attempts=3):
    """Calls fn() up to max_attempts times, returning the first successful
    result. fn should raise an exception to signal a failed attempt.
    Re-raises RetryExhaustedError if every attempt fails.

    Contract: max_attempts is the TOTAL number of attempts (including the
    first), and must be >= 1. This function is correct and already covered
    by its own tests -- do not modify it to "fix" a caller's bug.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_error = None
    for _ in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
    raise RetryExhaustedError(f"all {max_attempts} attempts failed") from last_error
