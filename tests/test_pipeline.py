from unittest.mock import Mock

import pytest

from contextselect.jev.base import Classifier
from contextselect.pipeline import SingleImagePipeline, SingleImageResult
from contextselect.schema import Context, Judgement
from contextselect.vision.base import ContextGenerator

IMAGE = b"image-bytes"
QUERY = "Which image shows a cutting tool?"
CONTEXT = Context(
    subject="kitchen knife",
    category="tool",
    appearance="A metal blade with a dark handle.",
    distinctive_features=["tapered blade", "handle"],
    function="Used for cutting food (inference).",
    state_condition="Intact.",
    action_context="No action is apparent.",
    environment="On a kitchen counter.",
    visible_text=[],
    related_concepts=["cutting tool"],
    uncertainty="",
)
JUDGEMENT = Judgement(score=0.91)


def make_pipeline() -> tuple[SingleImagePipeline[bytes], Mock, Mock]:
    context_generator = Mock(spec=ContextGenerator)
    context_generator.generate.return_value = CONTEXT
    classifier = Mock(spec=Classifier)
    classifier.score.return_value = JUDGEMENT
    return SingleImagePipeline(context_generator, classifier), context_generator, classifier


def test_successful_single_image_flow_returns_context_and_judgement() -> None:
    pipeline, context_generator, classifier = make_pipeline()

    result = pipeline.run(IMAGE, QUERY)

    assert isinstance(result, SingleImageResult)
    assert result.context is CONTEXT
    assert result.judgement is JUDGEMENT
    context_generator.generate.assert_called_once_with(IMAGE)
    classifier.score.assert_called_once_with(CONTEXT, QUERY)


def test_query_is_passed_unchanged_to_classifier() -> None:
    query_with_whitespace = "  " + QUERY + "  "
    pipeline, _, classifier = make_pipeline()

    pipeline.run(IMAGE, query_with_whitespace)

    classifier.score.assert_called_once_with(CONTEXT, query_with_whitespace)


def test_generated_context_is_passed_to_classifier() -> None:
    pipeline, context_generator, classifier = make_pipeline()

    pipeline.run(IMAGE, QUERY)

    generated_context = context_generator.generate.return_value
    classifier.score.assert_called_once_with(generated_context, QUERY)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_query_is_rejected_before_backend_calls(query: str) -> None:
    pipeline, context_generator, classifier = make_pipeline()

    with pytest.raises(ValueError, match="non-empty"):
        pipeline.run(IMAGE, query)

    context_generator.generate.assert_not_called()
    classifier.score.assert_not_called()


def test_vision_failure_is_propagated_unchanged() -> None:
    pipeline, context_generator, classifier = make_pipeline()
    error = RuntimeError("vision backend unavailable")
    context_generator.generate.side_effect = error

    with pytest.raises(RuntimeError, match="vision backend unavailable") as raised:
        pipeline.run(IMAGE, QUERY)

    assert raised.value is error
    classifier.score.assert_not_called()


def test_classifier_failure_is_propagated_unchanged() -> None:
    pipeline, _, classifier = make_pipeline()
    error = RuntimeError("Jev API unavailable")
    classifier.score.side_effect = error

    with pytest.raises(RuntimeError, match="Jev API unavailable") as raised:
        pipeline.run(IMAGE, QUERY)

    assert raised.value is error


def test_providers_are_injected_and_used() -> None:
    injected_context_generator = Mock(spec=ContextGenerator)
    injected_context_generator.generate.return_value = CONTEXT
    injected_classifier = Mock(spec=Classifier)
    injected_classifier.score.return_value = JUDGEMENT
    pipeline = SingleImagePipeline(injected_context_generator, injected_classifier)

    result = pipeline.run(IMAGE, QUERY)

    assert result.context is CONTEXT
    assert result.judgement is JUDGEMENT
    injected_context_generator.generate.assert_called_once_with(IMAGE)
    injected_classifier.score.assert_called_once_with(CONTEXT, QUERY)
