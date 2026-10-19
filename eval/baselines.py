"""Implemented and extension-point baselines for evaluation."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from contextselect.pipeline import SingleImagePipeline
from contextselect.ranker import ResultStatus
from contextselect.schema import Context

from eval.manifest import EvaluationTrial


@dataclass(frozen=True)
class ImageScore:
    """Score returned for an image, or None when a baseline has no score."""

    image_id: str
    score: float | None


@dataclass(frozen=True)
class BaselinePrediction:
    """Common prediction surface used by evaluation systems."""

    status: ResultStatus
    predicted_index: int | None
    image_scores: tuple[ImageScore, ...]


class EvaluationBaseline(Protocol):
    """Hook for a system that predicts one evaluation trial."""

    name: str
    requires_image_files: bool

    def predict(
        self,
        trial: EvaluationTrial,
        image_paths: Sequence[str],
        rng: random.Random,
    ) -> BaselinePrediction:
        """Predict a trial using its resolved image paths."""


class StructuredContextBaseline:
    """Use the proposed context -> Jev pipeline and configured ranker."""

    name = "structured_context_jev"
    requires_image_files = True

    def __init__(
        self,
        pipeline: SingleImagePipeline[str],
        confidence_threshold: float,
        ambiguity_margin: float,
    ) -> None:
        self._pipeline = pipeline
        self._confidence_threshold = confidence_threshold
        self._ambiguity_margin = ambiguity_margin

    def predict(
        self,
        trial: EvaluationTrial,
        image_paths: Sequence[str],
        rng: random.Random,
    ) -> BaselinePrediction:
        result = self._pipeline.run_many(
            image_paths,
            trial.query,
            confidence_threshold=self._confidence_threshold,
            ambiguity_margin=self._ambiguity_margin,
        )
        scores_by_index = {
            item.index: item.judgement.score for item in result.ranked_items
        }
        return BaselinePrediction(
            status=result.status,
            predicted_index=result.best_index,
            image_scores=tuple(
                ImageScore(image.id, scores_by_index[index])
                for index, image in enumerate(trial.images)
            ),
        )


class RandomBaseline:
    """Choose a random image for each trial; never predicts no-match."""

    name = "random"
    requires_image_files = False

    def predict(
        self,
        trial: EvaluationTrial,
        image_paths: Sequence[str],
        rng: random.Random,
    ) -> BaselinePrediction:
        return BaselinePrediction(
            status="best_match",
            predicted_index=rng.randrange(len(trial.images)),
            image_scores=tuple(ImageScore(image.id, None) for image in trial.images),
        )


UNIMPLEMENTED_BASELINES = {
    "direct_clip": "S1 direct CLIP-style image-text matching",
    "direct_vlm": "S2 direct VLM with all images",
    "caption_jev": "S3 free-form caption -> Jev",
    "structured_no_identity": "S5 structured context without subject/category",
}


class UnimplementedBaselineError(NotImplementedError):
    """Raised when selecting a documented baseline without an implementation."""


def require_implemented_baseline(name: str) -> None:
    """Fail clearly for documented baseline hooks not implemented in this phase."""
    if name in UNIMPLEMENTED_BASELINES:
        raise UnimplementedBaselineError(
            f"{UNIMPLEMENTED_BASELINES[name]} is an extension hook and is not implemented."
        )
