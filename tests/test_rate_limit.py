from datetime import datetime, timedelta, timezone

import pytest

from contextselect.rate_limit import SubmissionRateLimiter


def make_limiter(
    *,
    requests_per_minute: int = 5,
    daily_limit: int | None = 50,
    window_seconds: float = 60,
) -> tuple[SubmissionRateLimiter, list[float], list[datetime]]:
    monotonic_time = [0.0]
    utc_time = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    limiter = SubmissionRateLimiter(
        requests_per_minute=requests_per_minute,
        daily_limit=daily_limit,
        window_seconds=window_seconds,
        monotonic=lambda: monotonic_time[0],
        utcnow=lambda: utc_time[0],
    )
    return limiter, monotonic_time, utc_time


def test_allows_five_requests_per_minute_and_rejects_the_sixth() -> None:
    limiter, _, _ = make_limiter()

    for _ in range(5):
        assert limiter.check("client-a") is None

    message = limiter.check("client-a")

    assert message == (
        "Submission limit reached (5 per minute). Please wait before trying again."
    )


def test_minute_limit_uses_rolling_window_and_is_per_client() -> None:
    limiter, monotonic_time, _ = make_limiter()

    for _ in range(5):
        assert limiter.check("client-a") is None
    assert limiter.check("client-b") is None
    assert limiter.check("client-a") is not None

    monotonic_time[0] = 60
    assert limiter.check("client-a") is None


def test_daily_limit_counts_requests_across_minute_windows() -> None:
    limiter, monotonic_time, _ = make_limiter(
        requests_per_minute=100,
        daily_limit=2,
    )

    assert limiter.check("client-a") is None
    monotonic_time[0] = 61
    assert limiter.check("client-a") is None
    monotonic_time[0] = 122

    assert limiter.check("client-a") == (
        "Daily submission limit reached (2 per client). Try again after 00:00 UTC."
    )


def test_daily_limit_resets_at_utc_midnight() -> None:
    limiter, _, utc_time = make_limiter(daily_limit=1)

    assert limiter.check("client-a") is None
    assert limiter.check("client-a") is not None

    utc_time[0] += timedelta(days=1)
    assert limiter.check("client-a") is None


def test_daily_limit_can_be_disabled() -> None:
    limiter, monotonic_time, _ = make_limiter(
        requests_per_minute=1,
        daily_limit=None,
    )

    assert limiter.check("client-a") is None
    monotonic_time[0] = 60
    assert limiter.check("client-a") is None


@pytest.mark.parametrize(
    ("requests_per_minute", "daily_limit", "window_seconds"),
    [(0, 50, 60), (5, 0, 60), (5, None, 0)],
)
def test_invalid_limit_configuration_is_rejected(
    requests_per_minute: int,
    daily_limit: int | None,
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SubmissionRateLimiter(
            requests_per_minute=requests_per_minute,
            daily_limit=daily_limit,
            window_seconds=window_seconds,
        )
