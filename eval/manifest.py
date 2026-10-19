"""Evaluation dataset manifest models and loading."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManifestImage(BaseModel):
    """One local image reference in a manifest trial."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    path: str = Field(min_length=1)


class EvaluationTrial(BaseModel):
    """One query with one correct image or no matching image."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    images: list[ManifestImage] = Field(min_length=1, max_length=8)
    correct_index: int | None

    @model_validator(mode="after")
    def validate_correct_index_and_images(self) -> "EvaluationTrial":
        if not self.query.strip():
            raise ValueError("trial query must not be empty.")
        if self.correct_index is not None and not 0 <= self.correct_index < len(self.images):
            raise ValueError("correct_index must reference an image or be null for no-match.")
        image_ids = [image.id for image in self.images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image ids must be unique within a trial.")
        return self


class EvaluationManifest(BaseModel):
    """Versioned manifest describing reproducible N-way trials."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: int = Field(ge=1)
    default_n: int = Field(default=4, ge=1, le=8)
    trials: list[EvaluationTrial] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_trials(self) -> "EvaluationManifest":
        trial_ids = [trial.id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trial ids must be unique.")
        if any(len(trial.images) != self.default_n for trial in self.trials):
            raise ValueError(
                "every trial must contain exactly default_n images "
                f"({self.default_n})."
            )
        return self


def load_manifest(path: str | Path) -> EvaluationManifest:
    """Load and validate a JSON evaluation manifest."""
    manifest_path = Path(path)
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read evaluation manifest: {manifest_path}") from exc
    try:
        return EvaluationManifest.model_validate_json(manifest_text)
    except ValueError as exc:
        raise ValueError(f"Invalid evaluation manifest {manifest_path}: {exc}") from exc


def resolve_image_path(manifest_path: str | Path, image_path: str) -> Path:
    """Resolve an image path relative to its manifest unless it is absolute."""
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(manifest_path).resolve().parent / path
