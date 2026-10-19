import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from contextselect.jev.openrouter import (
    OPENROUTER_DECISIONS_ENDPOINT,
    OPENROUTER_JEV_MODEL,
    JevAnswerMissingError,
    JevConfigurationError,
    JevHTTPError,
    JevInvalidProbabilityError,
    JevMalformedResponseError,
    JevNetworkError,
    OpenRouterJevAdapter,
)
from contextselect.schema import Context, Judgement

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
VALID_RESPONSE = b'{"answers":{"match":{"type":"noul","noul":0.87}}}'


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
) -> tuple[OpenRouterJevAdapter, Mock, FakeResponse]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-token")
    response = FakeResponse(body, status)
    opener = Mock(return_value=response)
    return OpenRouterJevAdapter(opener=opener), opener, response


def test_successful_response_parses_probability_into_judgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, VALID_RESPONSE)

    assert adapter.score(VALID_CONTEXT, "a cutting tool") == Judgement(score=0.87)


def test_missing_api_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(JevConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterJevAdapter()


def test_http_error_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-token")

    def raise_http_error(*args: object, **kwargs: object) -> object:
        raise HTTPError(OPENROUTER_DECISIONS_ENDPOINT, 503, "Unavailable", {}, None)

    adapter = OpenRouterJevAdapter(opener=raise_http_error)

    with pytest.raises(JevHTTPError, match="503"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_network_error_raises_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-token")

    def raise_network_error(*args: object, **kwargs: object) -> object:
        raise URLError("connection failed")

    adapter = OpenRouterJevAdapter(opener=raise_network_error)

    with pytest.raises(JevNetworkError, match="Could not reach"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


@pytest.mark.parametrize("body", [b"{bad json", b"[]", b'"string"'])
def test_malformed_response_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, body)

    with pytest.raises(JevMalformedResponseError, match="malformed JSON|JSON object"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b'{"answers":null}',
        b'{"answers":{}}',
        b'{"answers":{"match":{}}}',
        b'{"answers":{"match":{"type":"noul"}}}',
    ],
)
def test_missing_decision_probability_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, body)

    with pytest.raises(JevAnswerMissingError, match="answers.match.noul"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


@pytest.mark.parametrize("probability", [1.1, -0.1, "0.5", True, float("inf"), float("nan")])
def test_invalid_probability_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    probability: object,
) -> None:
    response_body = json.dumps(
        {"answers": {"match": {"type": "noul", "noul": probability}}}
    ).encode()
    adapter, _, _ = make_adapter(monkeypatch, response_body)

    with pytest.raises(JevInvalidProbabilityError, match=r"\[0, 1\]"):
        adapter.score(VALID_CONTEXT, "a cutting tool")


def test_request_uses_openrouter_decisions_model_payload_and_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, opener, response = make_adapter(monkeypatch, VALID_RESPONSE)

    adapter.score(VALID_CONTEXT, "a cutting tool")

    request = opener.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == OPENROUTER_DECISIONS_ENDPOINT
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-openrouter-token"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert payload == {
        "model": "typesafe/jev-router",
        "state": VALID_CONTEXT.model_dump(mode="json"),
        "questions": {
            "match": {
                "type": "noul",
                "instructions": (
                    "Does the provided image context match the target concept "
                    "or query: a cutting tool?"
                ),
            }
        },
    }
    assert payload["model"] == OPENROUTER_JEV_MODEL
    assert opener.call_args.kwargs["timeout"] > 0
    assert response.closed


def test_non_success_status_raises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_adapter(monkeypatch, VALID_RESPONSE, status=429)

    with pytest.raises(JevHTTPError, match="429"):
        adapter.score(VALID_CONTEXT, "a cutting tool")
