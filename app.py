"""Gradio UI for query-conditioned image selection."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from contextselect.pipeline import SingleImagePipeline
from contextselect.ranker import MAX_IMAGES_PER_QUERY, RankedResult
from contextselect.jev.typesafe import (
    JevAdapterError,
    JevConfigurationError,
    TypeSafeJevAdapter,
)
from contextselect.vision.gemini import (
    GeminiConfigurationError,
    GeminiContextError,
    GeminiContextGenerator,
)

PipelineFactory = Callable[[], SingleImagePipeline[str]]


def create_pipeline() -> SingleImagePipeline[str]:
    """Construct the production providers, reading secrets from the environment."""
    return SingleImagePipeline(
        context_generator=GeminiContextGenerator(),
        classifier=TypeSafeJevAdapter(),
    )


def format_result(
    result: RankedResult,
    image_paths: Sequence[str],
) -> tuple[str, list[tuple[str, str]], list[dict[str, Any]]]:
    """Format pipeline output into status text, gallery entries, and context data."""
    status = f"**Status: `{result.status}`**"
    if result.status == "best_match" and result.best_index is not None:
        best_item = next(
            item for item in result.ranked_items if item.index == result.best_index
        )
        status += (
            f"\n\nBest match: image {result.best_index + 1} "
            f"(Jev score {best_item.judgement.score:.3f})."
        )
    elif result.status == "ambiguous":
        status += "\n\nNo unique best image: the leading scores are too close."
    else:
        status += "\n\nNo image met the confidence threshold."

    gallery: list[tuple[str, str]] = []
    contexts: list[dict[str, Any]] = []
    for item in result.ranked_items:
        image_path = image_paths[item.index]
        gallery.append(
            (
                image_path,
                f"Ranked {len(gallery) + 1} · Image {item.index + 1} · "
                f"Jev score {item.judgement.score:.3f}",
            )
        )
        contexts.append(
            {
                "rank": len(contexts) + 1,
                "image_index": item.index + 1,
                "image": Path(image_path).name,
                "jev_score": item.judgement.score,
                "context": item.context.model_dump(mode="json"),
            }
        )
    return status, gallery, contexts


def _safe_error_message(error: Exception) -> str:
    """Describe known provider failures without exposing credentials or raw SDK data."""
    if isinstance(error, GeminiConfigurationError):
        return "Gemini configuration is missing. Set GEMINI_API_KEY in the environment."
    if isinstance(error, JevConfigurationError):
        return "Jev configuration is missing. Set TYPESAFE_API_KEY in the environment."
    if isinstance(error, GeminiContextError):
        return f"Gemini context generation failed: {error}"
    if isinstance(error, JevAdapterError):
        return f"Jev scoring failed: {error}"
    return f"Processing failed ({type(error).__name__}). Check provider configuration and try again."


def handle_submission(
    image_paths: Sequence[str] | None,
    query: str | None,
    *,
    pipeline_factory: PipelineFactory | None = None,
) -> tuple[str, list[tuple[str, str]], list[dict[str, Any]]]:
    """Validate submission, invoke the pipeline, and format results for Gradio."""
    if not isinstance(query, str) or not query.strip():
        return "**Error:** Enter a non-empty query.", [], []
    if not image_paths:
        return "**Error:** Upload at least one image.", [], []
    if len(image_paths) > MAX_IMAGES_PER_QUERY:
        return f"**Error:** Upload no more than {MAX_IMAGES_PER_QUERY} images.", [], []

    paths = [str(path) for path in image_paths]
    try:
        pipeline = (pipeline_factory or create_pipeline)()
        result = pipeline.run_many(paths, query)
        return format_result(result, paths)
    except Exception as exc:
        return f"**Error:** {_safe_error_message(exc)}", [], []


def build_demo() -> gr.Blocks:
    """Build the minimal ContextSelect upload/query interface."""
    with gr.Blocks(title="ContextSelect") as demo:
        gr.Markdown(
            "# ContextSelect\n"
            "Select the image that best matches a concept using structured visual "
            "context and text-only relevance scoring."
        )
        with gr.Row():
            image_input = gr.File(
                label="Images (1–8)",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            query_input = gr.Textbox(
                label="Target concept or question",
                placeholder="Which image shows a cutting tool?",
                lines=2,
            )
        submit = gr.Button("Find best match", variant="primary")
        status_output = gr.Markdown()
        gallery_output = gr.Gallery(
            label="Ranked images",
            columns=4,
            object_fit="contain",
            height="auto",
        )
        context_output = gr.JSON(label="Generated context and Jev scores")
        submit.click(
            fn=handle_submission,
            inputs=[image_input, query_input],
            outputs=[status_output, gallery_output, context_output],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
