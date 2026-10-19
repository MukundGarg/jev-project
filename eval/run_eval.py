"""Run reproducible N-way image selection evaluation."""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from contextselect.pipeline import SingleImagePipeline
from contextselect.ranker import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from contextselect.jev.typesafe import TypeSafeJevAdapter
from contextselect.vision.gemini import GeminiContextGenerator
from eval.baselines import (
    BaselinePrediction,
    EvaluationBaseline,
    RandomBaseline,
    StructuredContextBaseline,
    require_implemented_baseline,
)
from eval.manifest import EvaluationManifest, EvaluationTrial, load_manifest, resolve_image_path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class TrialResult:
    """Recorded prediction and timing for one manifest trial."""

    trial_id: str
    query: str
    image_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    correct_index: int | None
    predicted_status: str
    predicted_index: int | None
    per_image_scores: tuple[dict[str, Any], ...]
    correct: bool
    latency_seconds: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate accuracy, no-match, and timing metrics."""

    trial_count: int
    top1_accuracy: float | None
    no_match_correct_reject_rate: float | None
    false_accept_rate: float | None
    average_latency_seconds: float


def _is_correct(trial: EvaluationTrial, prediction: BaselinePrediction) -> bool:
    if trial.correct_index is None:
        return prediction.status == "no_confident_match"
    return (
        prediction.status == "best_match"
        and prediction.predicted_index == trial.correct_index
    )


def _aggregate(results: list[TrialResult]) -> AggregateMetrics:
    normal_trials = [result for result in results if result.correct_index is not None]
    no_match_trials = [result for result in results if result.correct_index is None]
    correct_rejects = sum(
        result.predicted_status == "no_confident_match" for result in no_match_trials
    )
    false_accepts = sum(
        result.predicted_status != "no_confident_match" for result in no_match_trials
    )
    return AggregateMetrics(
        trial_count=len(results),
        top1_accuracy=(
            sum(result.correct for result in normal_trials) / len(normal_trials)
            if normal_trials
            else None
        ),
        no_match_correct_reject_rate=(
            correct_rejects / len(no_match_trials) if no_match_trials else None
        ),
        false_accept_rate=(
            false_accepts / len(no_match_trials) if no_match_trials else None
        ),
        average_latency_seconds=(
            sum(result.latency_seconds for result in results) / len(results)
            if results
            else 0.0
        ),
    )


def evaluate(
    manifest: EvaluationManifest,
    manifest_path: str | Path,
    baseline: EvaluationBaseline,
    *,
    seed: int = 0,
) -> tuple[list[TrialResult], AggregateMetrics]:
    """Run an injected baseline against every manifest trial."""
    rng = random.Random(seed)
    results: list[TrialResult] = []
    for trial in manifest.trials:
        resolved_paths = tuple(
            str(resolve_image_path(manifest_path, image.path)) for image in trial.images
        )
        if baseline.requires_image_files:
            for image_path in resolved_paths:
                if not Path(image_path).is_file():
                    raise FileNotFoundError(
                        f"Image file for trial {trial.id!r} does not exist: {image_path}"
                    )

        started = time.perf_counter()
        prediction = baseline.predict(trial, resolved_paths, rng)
        latency = time.perf_counter() - started
        if len(prediction.image_scores) != len(trial.images):
            raise ValueError(
                f"Baseline {baseline.name!r} returned a score list with the wrong length."
            )
        results.append(
            TrialResult(
                trial_id=trial.id,
                query=trial.query,
                image_ids=tuple(image.id for image in trial.images),
                image_paths=resolved_paths,
                correct_index=trial.correct_index,
                predicted_status=prediction.status,
                predicted_index=prediction.predicted_index,
                per_image_scores=tuple(
                    {"image_id": score.image_id, "score": score.score}
                    for score in prediction.image_scores
                ),
                correct=_is_correct(trial, prediction),
                latency_seconds=latency,
            )
        )

    return results, _aggregate(results)


def write_results(
    results: list[TrialResult],
    metrics: AggregateMetrics,
    *,
    baseline_name: str,
    seed: int,
    results_dir: str | Path,
) -> tuple[Path, Path]:
    """Write complete JSON and per-trial CSV result files."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{baseline_name}_results.json"
    csv_path = output_dir / f"{baseline_name}_trials.csv"
    json_payload = {
        "baseline": baseline_name,
        "seed": seed,
        "metrics": asdict(metrics),
        "trials": [asdict(result) for result in results],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "trial_id",
                "query",
                "image_ids",
                "image_paths",
                "correct_index",
                "predicted_status",
                "predicted_index",
                "per_image_scores",
                "correct",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for field in ("image_ids", "image_paths", "per_image_scores"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def _build_baseline(
    name: str,
    *,
    confidence_threshold: float,
    ambiguity_margin: float,
) -> EvaluationBaseline:
    require_implemented_baseline(name)
    if name == "random":
        return RandomBaseline()
    if name == "structured_context_jev":
        pipeline = SingleImagePipeline(
            GeminiContextGenerator(),
            TypeSafeJevAdapter(),
        )
        return StructuredContextBaseline(
            pipeline,
            confidence_threshold=confidence_threshold,
            ambiguity_margin=ambiguity_margin,
        )
    raise ValueError(f"Unknown evaluation system: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--system",
        default="structured_context_jev",
        choices=[
            "structured_context_jev",
            "random",
            "direct_clip",
            "direct_vlm",
            "caption_jev",
            "structured_no_identity",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=DEFAULT_AMBIGUITY_MARGIN,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        baseline = _build_baseline(
            args.system,
            confidence_threshold=args.confidence_threshold,
            ambiguity_margin=args.ambiguity_margin,
        )
        results, metrics = evaluate(
            manifest,
            args.manifest,
            baseline,
            seed=args.seed,
        )
        json_path, csv_path = write_results(
            results,
            metrics,
            baseline_name=baseline.name,
            seed=args.seed,
            results_dir=args.results_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(metrics), indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
