import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import Mock

import pytest

from eval.baselines import (
    BaselinePrediction,
    ImageScore,
    RandomBaseline,
    StructuredContextBaseline,
    UnimplementedBaselineError,
    require_implemented_baseline,
)
from eval.manifest import EvaluationManifest, EvaluationTrial, ManifestImage
from eval.run_eval import evaluate, write_results
from contextselect.ranker import RankedItem, RankedResult
from contextselect.schema import Context, Judgement


def make_trial(
    trial_id: str,
    correct_index: int | None,
) -> EvaluationTrial:
    return EvaluationTrial(
        id=trial_id,
        query="Find the target item",
        correct_index=correct_index,
        images=[
            ManifestImage(id=f"{trial_id}-{index}", path=f"{index}.jpg")
            for index in range(4)
        ],
    )


def make_manifest() -> EvaluationManifest:
    return EvaluationManifest(
        version=1,
        default_n=4,
        trials=[
            make_trial("normal", 2),
            make_trial("no-match", None),
        ],
    )


def touch_manifest_images(
    tmp_path: Path,
    manifest: EvaluationManifest,
) -> Path:
    for trial in manifest.trials:
        for image in trial.images:
            (tmp_path / image.path).touch()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest_path


def test_manifest_accepts_normal_and_no_match_trials() -> None:
    manifest = make_manifest()

    assert manifest.default_n == 4
    assert manifest.trials[0].correct_index == 2
    assert manifest.trials[1].correct_index is None


def test_manifest_rejects_wrong_trial_size() -> None:
    with pytest.raises(ValueError, match="exactly default_n"):
        EvaluationManifest(
            version=1,
            default_n=4,
            trials=[
                EvaluationTrial(
                    id="bad",
                    query="target",
                    correct_index=0,
                    images=[ManifestImage(id="one", path="one.jpg")],
                )
            ],
        )


def test_random_baseline_is_seeded_and_never_rejects() -> None:
    trial = make_trial("trial", None)
    baseline = RandomBaseline()

    first = baseline.predict(trial, ["a", "b", "c", "d"], random.Random(11))
    second = baseline.predict(trial, ["a", "b", "c", "d"], random.Random(11))

    assert first == second
    assert first.status == "best_match"
    assert first.predicted_index in range(4)
    assert [score.score for score in first.image_scores] == [None] * 4


def test_structured_baseline_adapts_pipeline_ranked_output() -> None:
    context = Context(
        subject="tool",
        category="tool",
        appearance="Metal object.",
        distinctive_features=[],
        function="Unclear.",
        state_condition="Intact.",
        action_context="No action.",
        environment="Plain background.",
        visible_text=[],
        related_concepts=[],
        uncertainty="",
    )
    pipeline = Mock()
    pipeline.run_many.return_value = RankedResult(
        ranked_items=(
            RankedItem(index=1, context=context, judgement=Judgement(score=0.9)),
            RankedItem(index=0, context=context, judgement=Judgement(score=0.4)),
            RankedItem(index=2, context=context, judgement=Judgement(score=0.2)),
            RankedItem(index=3, context=context, judgement=Judgement(score=0.1)),
        ),
        status="best_match",
        best_index=1,
    )
    baseline = StructuredContextBaseline(pipeline, 0.5, 0.05)
    trial = make_trial("structured", 1)

    prediction = baseline.predict(
        trial,
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        random.Random(0),
    )

    assert prediction.status == "best_match"
    assert prediction.predicted_index == 1
    assert [entry.image_id for entry in prediction.image_scores] == [
        image.id for image in trial.images
    ]
    assert [entry.score for entry in prediction.image_scores] == [
        0.4,
        0.9,
        0.2,
        0.1,
    ]
    pipeline.run_many.assert_called_once_with(
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        trial.query,
        confidence_threshold=0.5,
        ambiguity_margin=0.05,
    )


def test_evaluation_calculates_metrics_and_trial_records(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest_path = touch_manifest_images(tmp_path, manifest)
    baseline = Mock()
    baseline.name = "mock_structured"
    baseline.predict.side_effect = [
        BaselinePrediction(
            status="best_match",
            predicted_index=2,
            image_scores=tuple(
                ImageScore(f"normal-{index}", 0.1 * index) for index in range(4)
            ),
        ),
        BaselinePrediction(
            status="no_confident_match",
            predicted_index=None,
            image_scores=tuple(
                ImageScore(f"no-match-{index}", 0.2) for index in range(4)
            ),
        ),
    ]

    results, metrics = evaluate(
        manifest,
        manifest_path,
        baseline,
        seed=7,
    )

    assert len(results) == 2
    assert results[0].correct is True
    assert results[0].correct_index == 2
    assert results[0].predicted_index == 2
    assert results[0].per_image_scores[2]["score"] == 0.2
    assert results[1].correct is True
    assert metrics.trial_count == 2
    assert metrics.top1_accuracy == 1.0
    assert metrics.no_match_correct_reject_rate == 1.0
    assert metrics.false_accept_rate == 0.0
    assert metrics.average_latency_seconds >= 0.0


def test_false_accept_is_counted_for_no_match_trial(tmp_path: Path) -> None:
    manifest = EvaluationManifest(
        version=1,
        default_n=4,
        trials=[make_trial("no-match", None)],
    )
    manifest_path = touch_manifest_images(tmp_path, manifest)
    baseline = Mock()
    baseline.name = "mock"
    baseline.predict.return_value = BaselinePrediction(
        status="best_match",
        predicted_index=0,
        image_scores=tuple(
            ImageScore(f"no-match-{index}", 0.9) for index in range(4)
        ),
    )

    _, metrics = evaluate(manifest, manifest_path, baseline)

    assert metrics.no_match_correct_reject_rate == 0.0
    assert metrics.false_accept_rate == 1.0


def test_evaluation_fails_explicitly_for_missing_image(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    baseline = Mock()
    baseline.name = "image-consuming"
    baseline.requires_image_files = True

    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluate(manifest, manifest_path, baseline)


def test_random_baseline_does_not_need_demo_image_files(tmp_path: Path) -> None:
    manifest_path = Path(__file__).parents[1] / "eval" / "examples" / "demo_manifest.json"

    results, metrics = evaluate(
        EvaluationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        ),
        manifest_path,
        RandomBaseline(),
        seed=0,
    )

    assert len(results) == 2
    assert metrics.trial_count == 2
    assert all(result.predicted_status == "best_match" for result in results)


def test_module_command_writes_results_from_demo_manifest(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    manifest_path = project_root / "eval" / "examples" / "demo_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.run_eval",
            "--manifest",
            str(manifest_path),
            "--system",
            "random",
            "--seed",
            "5",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "random_trials.csv").is_file()
    assert (tmp_path / "random_results.json").is_file()


def test_write_results_creates_json_and_csv(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest_path = touch_manifest_images(tmp_path, manifest)
    baseline = Mock()
    baseline.name = "mock"
    baseline.predict.side_effect = [
        BaselinePrediction("best_match", 2, tuple(
            ImageScore(f"normal-{index}", 0.5) for index in range(4)
        )),
        BaselinePrediction("no_confident_match", None, tuple(
            ImageScore(f"no-match-{index}", 0.1) for index in range(4)
        )),
    ]
    trial_results, metrics = evaluate(manifest, manifest_path, baseline, seed=3)

    json_path, csv_path = write_results(
        trial_results,
        metrics,
        baseline_name=baseline.name,
        seed=3,
        results_dir=tmp_path / "results",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 3
    assert payload["metrics"]["trial_count"] == 2
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert json.loads(rows[0]["per_image_scores"])[0]["image_id"] == "normal-0"


def test_unavailable_baseline_hook_fails_clearly() -> None:
    with pytest.raises(UnimplementedBaselineError, match="extension hook"):
        require_implemented_baseline("direct_clip")
