"""Cache interface; storage behavior is implemented in a later phase."""

from abc import ABC, abstractmethod

from contextselect.schema import Context


class ContextCache(ABC):
    """Load and store generated contexts by a caller-provided cache key."""

    @abstractmethod
    def get(self, key: str) -> Context | None:
        """Return a cached context, or None on a cache miss."""

    @abstractmethod
    def set(self, key: str, context: Context) -> None:
        """Store a generated context under its cache key."""

