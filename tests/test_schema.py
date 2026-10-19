from pydantic import ValidationError
import pytest

from contextselect.schema import Context, Judgement


def test_context_requires_documented_fields() -> None:
    with pytest.raises(ValidationError):
        Context(subject="mug")


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_judgement_rejects_scores_outside_probability_range(score: float) -> None:
    with pytest.raises(ValidationError):
        Judgement(score=score)
