"""Small in-memory submission limits for the demo UI."""

from collections import defaultdict, deque
from collections.abc import Callable
from datetime import date, datetime, timezone
import time
from threading import Lock


class SubmissionRateLimiter:
    """Limit submissions per client using a rolling minute and UTC-day counter."""

    def __init__(
        self,
        *,
        requests_per_minute: int = 5,
        daily_limit: int | None = 50,
        window_seconds: float = 60,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive.")
        if daily_limit is not None and daily_limit < 1:
            raise ValueError("daily_limit must be positive or None.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        self._requests_per_minute = requests_per_minute
        self._daily_limit = daily_limit
        self._window_seconds = window_seconds
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._minute_requests: dict[str, deque[float]] = defaultdict(deque)
        self._daily_requests: dict[str, tuple[date, int]] = {}
        self._lock = Lock()

    def check(self, client_id: str) -> str | None:
        """Record one submission, returning a user-facing denial message if limited."""
        now = self._monotonic()
        today = self._utcnow().astimezone(timezone.utc).date()

        with self._lock:
            requests = self._minute_requests[client_id]
            while requests and requests[0] <= now - self._window_seconds:
                requests.popleft()

            daily_date, daily_count = self._daily_requests.get(client_id, (today, 0))
            if self._daily_limit is not None and daily_date == today:
                if daily_count >= self._daily_limit:
                    return (
                        f"Daily submission limit reached ({self._daily_limit} per client). "
                        "Try again after 00:00 UTC."
                    )
            if len(requests) >= self._requests_per_minute:
                return (
                    f"Submission limit reached ({self._requests_per_minute} per minute). "
                    "Please wait before trying again."
                )

            requests.append(now)
            self._daily_requests[client_id] = (
                (today, daily_count + 1) if daily_date == today else (today, 1)
            )
            return None
