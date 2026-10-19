import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from contextselect.schema import Context, Judgement
from contextselect.jev.typesafe import (
    JEV_ENDPOINT,
    JevAnswerMissingError,
    JevConfigurationError,
    JevHTTPError,
    JevInvalidProbabilityError,
    JevMalformedResponseError,
    JevNetworkError,
    TypeSafeJevAdapter,
)

VALID_CONTEXT = Context(
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


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.closed = False

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        self.closed = True


def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    status: int = 200,
) -> tuple[TypeSafeJevAdapter, Mock, FakeResponse]:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-token")
    response = FakeResponse(body, status)
    opener = Mock(return_value=response)
    return TypeSafeJevAdapter(opener=opener), opener, response


def test_successful_response_returns_judgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(
        monkeypatch, b'{"answers":{"match":{"noul":0.87}}}'
    )

    result = adapter.score(VALID_CONTEXT, "a cutting tool")

    assert result == Judgement(score=0.87)


def test_missing_api_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(JevConfigurationError, match="TYPESAFE_API_KEY"):
        TypeSafeJevAdapter()


def test_http_error_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-token")

    def raise_http_error(*args: object, **kwargs: object) -> object:
        raise HTTPError(JEV_ENDPOINT, 503, "Unavailable", {}, None)

    adapter = TypeSafeJevAdapter(opener=raise_http_error)

    with pytest.raises(JevHTTPError, match="503"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_network_error_raises_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-token")

    def raise_network_error(*args: object, **kwargs: object) -> object:
        raise URLError("connection failed")

    adapter = TypeSafeJevAdapter(opener=raise_network_error)

    with pytest.raises(JevNetworkError, match="Could not reach"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_malformed_json_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, b"{bad json")

    with pytest.raises(JevMalformedResponseError, match="malformed JSON"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_missing_noul_answer_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, b'{"answers":{"match":{}}}')

    with pytest.raises(JevAnswerMissingError, match="answers.match.noul"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


@pytest.mark.parametrize("probability", [1.1, -0.1, "0.5", True, float("inf")])
def test_invalid_probability_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    probability: object,
) -> None:
    response_body = json.dumps({"answers": {"match": {"noul": probability}}}).encode()
    adapter, _, _ = make_adapter(monkeypatch, response_body)

    with pytest.raises(JevInvalidProbabilityError, match=r"\[0, 1\]"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_request_payload_and_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, opener, response = make_adapter(
        monkeypatch, b'{"answers":{"match":{"noul":0.6}}}'
    )

    adapter.score(VALID_CONTEXT, "a cutting tool")

    request = opener.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == JEV_ENDPOINT
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert payload == {
        "state": VALID_CONTEXT.model_dump(mode="json"),
        "questions": {
            "match": {
                "type": "noul",
                "instructions": (
                    "Does the provided image context match the target concept: "
                    "a cutting tool?"
                ),
            }
        },
    }
    assert opener.call_args.kwargs["timeout"] > 0
    assert response.closed


def test_non_success_status_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(
        monkeypatch, b'{"answers":{"match":{"noul":0.6}}}', status=429
    )

    with pytest.raises(JevHTTPError, match="429"):
        adapter.score(VALID_CONTEXT, "a cutting tool")
