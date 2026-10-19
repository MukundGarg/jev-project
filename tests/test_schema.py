import pytest
from pydantic import ValidationError

from contextselect.schema import Context, Judgement


def make_valid_context() -> Context:
    return Context(
        subject="ceramic mug",
        category="drinkware",
        appearance="A small white vessel with a curved handle.",
        distinctive_features=["curved handle"],
        function="Used for holding a drink (inference).",
        state_condition="Intact; no visible damage.",
        action_context="No action is apparent.",
        environment="On a wooden table.",
        visible_text=[],
        related_concepts=["cup"],
        uncertainty="",
    )


def test_context_accepts_valid_documented_fields() -> None:
    context = make_valid_context()

    assert context.subject == "ceramic mug"
    assert context.distinctive_features == ["curved handle"]
    assert context.visible_text == []


def test_context_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Context(subject="mug")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject", 42),
        ("distinctive_features", "curved handle"),
        ("visible_text", [123]),
        ("related_concepts", None),
    ],
)
def test_context_rejects_wrong_field_types(field: str, value: object) -> None:
    values = make_valid_context().model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        Context.model_validate(values)


def test_context_serializes_to_dict_and_json() -> None:
    context = make_valid_context()

    data = context.model_dump()
    serialized = context.model_dump_json()

    assert data["subject"] == "ceramic mug"
    assert data["distinctive_features"] == ["curved handle"]
    assert '"category":"drinkware"' in serialized


def test_judgement_accepts_probability() -> None:
    assert Judgement(score=0.75).score == 0.75


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_judgement_rejects_scores_outside_probability_range(score: float) -> None:
    with pytest.raises(ValidationError):
        Judgement(score=score)
