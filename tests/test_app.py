from unittest.mock import Mock, patch

import gradio as gr
import pytest

import app
from contextselect.pipeline import SingleImagePipeline
from contextselect.ranker import RankedItem, RankedResult
from contextselect.schema import Context, Judgement
from contextselect.vision.gemini import GeminiConfigurationError
from contextselect.jev.typesafe import JevConfigurationError


def make_context(subject: str) -> Context:
    return Context(
        subject=subject,
        category="tool",
        appearance="Metal object.",
        distinctive_features=["handle"],
        function="Used for cutting (inference).",
        state_condition="Intact.",
        action_context="No action is apparent.",
        environment="On a work surface.",
        visible_text=[],
        related_concepts=[],
        uncertainty="",
    )


def make_result() -> RankedResult:
    return RankedResult(
        ranked_items=(
            RankedItem(1, make_context("knife"), Judgement(score=0.91)),
            RankedItem(0, make_context("scissors"), Judgement(score=0.6)),
        ),
        status="best_match",
        best_index=1,
    )


def test_create_pipeline_injects_provider_instances() -> None:
    context_generator = Mock()
    classifier = Mock()
    with (
        patch.object(app, "GeminiContextGenerator", return_value=context_generator),
        patch.object(app, "TypeSafeJevAdapter", return_value=classifier),
    ):
        pipeline = app.create_pipeline()

    assert isinstance(pipeline, SingleImagePipeline)
    assert pipeline._context_generator is context_generator
    assert pipeline._classifier is classifier


def test_submit_handler_runs_pipeline_with_uploaded_paths_and_query() -> None:
    pipeline = Mock()
    pipeline.run_many.return_value = make_result()
    image_paths = ["/tmp/first.png", "/tmp/second.jpg"]
    factory = Mock(return_value=pipeline)

    status, gallery, contexts = app.handle_submission(
        image_paths,
        "Which image shows a cutting tool?",
        pipeline_factory=factory,
    )

    factory.assert_called_once_with()
    pipeline.run_many.assert_called_once_with(
        image_paths,
        "Which image shows a cutting tool?",
    )
    assert "best_match" in status
    assert "image 2" in status
    assert gallery == [
        ("/tmp/second.jpg", "Ranked 1 · Image 2 · Jev score 0.910"),
        ("/tmp/first.png", "Ranked 2 · Image 1 · Jev score 0.600"),
    ]
    assert contexts[0]["image_index"] == 2
    assert contexts[0]["context"]["subject"] == "knife"
    assert contexts[0]["jev_score"] == 0.91


@pytest.mark.parametrize(
    ("image_paths", "query", "expected"),
    [
        (["/tmp/a.png"], "  ", "Enter a non-empty query"),
        ([], "find tool", "Upload at least one image"),
        ([f"/tmp/{index}.png" for index in range(9)], "find tool", "no more than 8"),
        (None, "find tool", "Upload at least one image"),
    ],
)
def test_submit_handler_rejects_invalid_input_without_constructing_provider(
    image_paths: list[str] | None,
    query: str,
    expected: str,
) -> None:
    factory = Mock()

    status, gallery, contexts = app.handle_submission(
        image_paths,
        query,
        pipeline_factory=factory,
    )

    assert expected in status
    assert gallery == []
    assert contexts == []
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (GeminiConfigurationError("GEMINI_API_KEY required"), "GEMINI_API_KEY"),
        (JevConfigurationError("TYPESAFE_API_KEY required"), "TYPESAFE_API_KEY"),
        (RuntimeError("secret must not leak"), "RuntimeError"),
    ],
)
def test_submit_handler_surfaces_safe_provider_errors(
    error: Exception,
    expected: str,
) -> None:
    factory = Mock(side_effect=error)

    status, gallery, contexts = app.handle_submission(
        ["/tmp/image.png"],
        "find the target",
        pipeline_factory=factory,
    )

    assert expected in status
    if isinstance(error, RuntimeError) and not isinstance(error, (GeminiConfigurationError, JevConfigurationError)):
        assert "secret must not leak" not in status
    assert gallery == []
    assert contexts == []


def test_format_result_displays_ambiguous_and_no_match_statuses() -> None:
    ambiguous = RankedResult(
        ranked_items=(
            RankedItem(0, make_context("one"), Judgement(score=0.8)),
            RankedItem(1, make_context("two"), Judgement(score=0.78)),
        ),
        status="ambiguous",
        best_index=None,
    )
    no_match = RankedResult(
        ranked_items=(RankedItem(0, make_context("one"), Judgement(score=0.2)),),
        status="no_confident_match",
        best_index=None,
    )

    ambiguous_status, _, _ = app.format_result(ambiguous, ["/tmp/a.png", "/tmp/b.png"])
    no_match_status, _, _ = app.format_result(no_match, ["/tmp/a.png"])

    assert "ambiguous" in ambiguous_status
    assert "No unique best image" in ambiguous_status
    assert "no_confident_match" in no_match_status
    assert "No image met the confidence threshold" in no_match_status


def test_build_demo_returns_gradio_blocks() -> None:
    demo = app.build_demo()

    assert isinstance(demo, gr.Blocks)
