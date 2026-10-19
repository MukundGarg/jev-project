"""Single-image orchestration for context generation and relevance scoring."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from contextselect.jev.base import Classifier
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
