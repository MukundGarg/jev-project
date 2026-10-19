"""Interface for query-agnostic image context generators."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from contextselect.schema import Context

ImageT = TypeVar("ImageT")


class ContextGenerator(ABC, Generic[ImageT]):
    """Generate structured semantic context from one image."""

    @abstractmethod
    def generate(self, image: ImageT) -> Context:
        """Describe the image without using a user query."""
