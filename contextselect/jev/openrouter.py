"""OpenRouter Decisions API adapter for TypeSafe Jev."""

import json
import math
import os
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from contextselect.jev.base import Classifier
from contextselect.schema import Context, Judgement

OPENROUTER_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "typesafe/jev-router"
DEFAULT_TIMEOUT_SECONDS = 30.0


class JevAdapterError(RuntimeError):
    """Base exception for Jev adapter failures."""


class JevConfigurationError(JevAdapterError):
    """Raised when the OpenRouter API key is missing."""


class JevNetworkError(JevAdapterError):
    """Raised when a request cannot reach OpenRouter."""


class JevHTTPError(JevAdapterError):
    """Raised when OpenRouter returns a non-success HTTP status."""


class JevMalformedResponseError(JevAdapterError):
    """Raised when the response body is not a valid decision JSON object."""


class JevAnswerMissingError(JevAdapterError):
    """Raised when the response does not contain the requested decision."""


class JevInvalidProbabilityError(JevAdapterError):
    """Raised when the Jev decision is not a probability in [0, 1]."""


class _Response(Protocol):
    status: int

    def read(self) -> bytes: ...

    def close(self) -> None: ...


Opener = Callable[..., _Response]


def _parse_probability(response_body: bytes) -> float:
    try:
        payload = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise JevMalformedResponseError(
            "OpenRouter Decisions API returned malformed JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise JevMalformedResponseError(
            "OpenRouter Decisions API response must be a JSON object."
        )

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevAnswerMissingError(
            "OpenRouter decision response is missing answers.match.noul."
        )
    match_answer = answers.get("match")
    if not isinstance(match_answer, dict) or "noul" not in match_answer:
        raise JevAnswerMissingError(
            "OpenRouter decision response is missing answers.match.noul."
        )

    probability = match_answer["noul"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise JevInvalidProbabilityError(
            "OpenRouter answers.match.noul must be a finite number in [0, 1]."
        )

    return float(probability)


class OpenRouterJevAdapter(Classifier):
    """Score structured textual context against a query using Jev via OpenRouter."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Opener | None = None,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise JevConfigurationError(
                "OPENROUTER_API_KEY is required to use OpenRouterJevAdapter."
            )

        self._api_key = api_key
        self._timeout = timeout
        self._opener = opener or urlopen

    def score(self, context: Context, query: str) -> Judgement:
        request_payload = {
            "model": OPENROUTER_JEV_MODEL,
            "state": context.model_dump(mode="json"),
            "questions": {
                "match": {
                    "type": "noul",
                    "instructions": (
                        "Does the provided image context match the target concept "
                        "or query: "
                        f"{query}?"
                    ),
                }
            },
        }
        request = Request(
            OPENROUTER_DECISIONS_ENDPOINT,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            response = self._opener(request, timeout=self._timeout)
            try:
                status = response.status
                response_body = response.read()
            finally:
                response.close()
        except HTTPError as exc:
            raise JevHTTPError(
                f"OpenRouter returned HTTP status {exc.code}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise JevNetworkError("Could not reach the OpenRouter Decisions API.") from exc

        if not 200 <= status < 300:
            raise JevHTTPError(f"OpenRouter returned HTTP status {status}.")

        return Judgement(score=_parse_probability(response_body))
