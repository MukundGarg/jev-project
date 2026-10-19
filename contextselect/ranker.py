"""Ranker interface; selection logic is implemented in a later phase."""

from collections.abc import Sequence
from typing import Protocol

from contextselect.schema import Judgement


class Ranker(Protocol):
    """Order classifier judgements using their relevance scores."""

    def rank(
        self, judgements: Sequence[Judgement], threshold: float
    ) -> "RankedResult":
        """Return ordered judgements and the best match, if any."""
        ...


class RankedResult(Protocol):
    """Result shape for a future ranker implementation."""

    ordered: Sequence[Judgement]
    best: Judgement | None
