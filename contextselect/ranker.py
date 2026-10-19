"""Rank per-image classifier results and decide match confidence."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from contextselect.schema import Context, Judgement

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_AMBIGUITY_MARGIN = 0.05
MAX_IMAGES_PER_QUERY = 8

ResultStatus = Literal["best_match", "no_confident_match", "ambiguous"]


@dataclass(frozen=True)
class RankedItem:
    """Context and classifier judgement, retaining the input image index."""

    index: int
    context: Context
    judgement: Judgement


@dataclass(frozen=True)
class RankedResult:
    """Ranked image results and the overall selection status."""

    ranked_items: tuple[RankedItem, ...]
    status: ResultStatus
    best_index: int | None


def validate_rank_options(confidence_threshold: float, ambiguity_margin: float) -> None:
    """Validate score options before any image-provider calls are made."""
    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be a finite number in [0, 1].")
    if (
        isinstance(ambiguity_margin, bool)
        or not isinstance(ambiguity_margin, (int, float))
        or not math.isfinite(ambiguity_margin)
        or not 0.0 <= ambiguity_margin <= 1.0
    ):
        raise ValueError("ambiguity_margin must be a finite number in [0, 1].")


def rank(
    items: Sequence[RankedItem],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> RankedResult:
    """Rank results by score, stably preserving input order for ties.

    Scores below the confidence threshold yield ``no_confident_match``.
    Otherwise, a top-two score difference less than or equal to the ambiguity
    margin yields ``ambiguous``. ``best_index`` is set only for an unambiguous
    confident match.
    """
    if not items:
        raise ValueError("At least one ranked item is required.")
    validate_rank_options(confidence_threshold, ambiguity_margin)

    ranked_items = tuple(
        sorted(items, key=lambda item: item.judgement.score, reverse=True)
    )
    top_item = ranked_items[0]

    if top_item.judgement.score < confidence_threshold:
        return RankedResult(
            ranked_items=ranked_items,
            status="no_confident_match",
            best_index=None,
        )

    if (
        len(ranked_items) > 1
        and top_item.judgement.score - ranked_items[1].judgement.score
        <= ambiguity_margin
    ):
        return RankedResult(
            ranked_items=ranked_items,
            status="ambiguous",
            best_index=None,
        )

    return RankedResult(
        ranked_items=ranked_items,
        status="best_match",
        best_index=top_item.index,
    )
