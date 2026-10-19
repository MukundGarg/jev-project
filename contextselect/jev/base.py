"""Interface for text-only context classifiers."""

from abc import ABC, abstractmethod

from contextselect.schema import Context, Judgement


class Classifier(ABC):
    """Score one semantic context against a user query."""

    @abstractmethod
    def score(self, context: Context, query: str) -> Judgement:
        """Return the context's relevance probability for the query."""

