import pytest

from contextselect.ranker import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_CONFIDENCE_THRESHOLD,
    RankedItem,
    rank,
)
from contextselect.schema import Context, Judgement


def make_context(subject: str) -> Context:
    return Context(
        subject=subject,
        category="tool",
        appearance="A visible object.",
        distinctive_features=[],
        function="Unclear.",
        state_condition="No visible damage.",
        action_context="No action is apparent.",
        environment="A plain background.",
        visible_text=[],
        related_concepts=[],
        uncertainty="",
    )


def make_item(index: int, score: float) -> RankedItem:
    return RankedItem(
        index=index,
        context=make_context(f"object {index}"),
        judgement=Judgement(score=score),
    )


def test_normal_ranking_sorts_descending_and_selects_top_match() -> None:
    result = rank([make_item(0, 0.62), make_item(1, 0.91), make_item(2, 0.2)])

    assert [item.index for item in result.ranked_items] == [1, 0, 2]
    assert result.status == "best_match"
    assert result.best_index == 1


def test_no_confident_match_when_top_score_is_below_threshold() -> None:
    result = rank(
        [make_item(0, 0.48), make_item(1, 0.2)],
        confidence_threshold=0.5,
    )

    assert result.status == "no_confident_match"
    assert result.best_index is None
    assert result.ranked_items[0].index == 0


def test_near_tie_is_ambiguous_without_best_index() -> None:
    result = rank(
        [make_item(0, 0.86), make_item(1, 0.83)],
        ambiguity_margin=0.05,
    )

    assert result.status == "ambiguous"
    assert result.best_index is None
    assert [item.index for item in result.ranked_items] == [0, 1]


def test_equal_scores_preserve_input_order() -> None:
    result = rank([make_item(4, 0.9), make_item(2, 0.9), make_item(7, 0.5)])

    assert [item.index for item in result.ranked_items] == [4, 2, 7]
    assert result.status == "ambiguous"


def test_single_item_is_a_best_match_when_confident() -> None:
    result = rank([make_item(3, 0.8)])

    assert result.status == "best_match"
    assert result.best_index == 3


def test_rank_defaults_are_documented_constants() -> None:
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.5
    assert DEFAULT_AMBIGUITY_MARGIN == 0.05


@pytest.mark.parametrize(
    ("threshold", "margin"),
    [
        (-0.1, 0.05),
        (0.5, 1.1),
        (float("nan"), 0.05),
    ],
)
def test_invalid_ranking_options_are_rejected(
    threshold: float, margin: float
) -> None:
    with pytest.raises(ValueError):
        rank([make_item(0, 0.8)], threshold, margin)


def test_empty_ranked_items_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        rank([])
