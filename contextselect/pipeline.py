"""Single-image orchestration for context generation and relevance scoring."""

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

from contextselect.jev.base import Classifier
from contextselect.ranker import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_IMAGES_PER_QUERY,
    RankedItem,
    RankedResult,
    rank,
    validate_rank_options,
)
from contextselect.schema import Context, Judgement
from contextselect.vision.base import ContextGenerator

ImageT = TypeVar("ImageT")


@dataclass(frozen=True)
class SingleImageResult:
    """Context and Jev judgement produced for one image and query."""

    context: Context
    judgement: Judgement


class SingleImagePipeline(Generic[ImageT]):
    """Generate one image context, then score it against one query."""

    def __init__(
        self,
        context_generator: ContextGenerator[ImageT],
        classifier: Classifier,
    ) -> None:
        self._context_generator = context_generator
        self._classifier = classifier

    def run(self, image: ImageT, query: str) -> SingleImageResult:
        """Process one image without changing the supplied query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        context = self._context_generator.generate(image)
        judgement = self._classifier.score(context, query)
        return SingleImageResult(context=context, judgement=judgement)

    def run_many(
        self,
        images: Sequence[ImageT],
        query: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    ) -> RankedResult:
        """Score and rank 1-8 images in input order using injected providers."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(images, (str, bytes, bytearray, Path)):
            raise TypeError("images must be a sequence of image inputs.")
        if not images:
            raise ValueError("At least one image is required.")
        if len(images) > MAX_IMAGES_PER_QUERY:
            raise ValueError(f"A maximum of {MAX_IMAGES_PER_QUERY} images is allowed.")
        validate_rank_options(confidence_threshold, ambiguity_margin)

        items: list[RankedItem] = []
        for index, image in enumerate(images):
            context = self._context_generator.generate(image)
            judgement = self._classifier.score(context, query)
            items.append(
                RankedItem(index=index, context=context, judgement=judgement)
            )

        return rank(items, confidence_threshold, ambiguity_margin)
