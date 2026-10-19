"""TypeSafe Jev API adapter."""

import json
import math
import os
from http.client import HTTPResponse
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from contextselect.schema import Context, Judgement
from contextselect.jev.base import Classifier

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_TIMEOUT_SECONDS = 30.0


class JevAdapterError(RuntimeError):
    """Base exception for TypeSafe Jev adapter failures."""


class JevConfigurationError(JevAdapterError):
    """Raised when the TypeSafe API token is missing."""


class JevNetworkError(JevAdapterError):
    """Raised when a request cannot reach the TypeSafe API."""


class JevHTTPError(JevAdapterError):
    """Raised when TypeSafe returns a non-success HTTP status."""


class JevMalformedResponseError(JevAdapterError):
    """Raised when the response body is not valid JSON."""


class JevAnswerMissingError(JevAdapterError):
    """Raised when answers.match.noul is absent from the response."""


class JevInvalidProbabilityError(JevAdapterError):
    """Raised when answers.match.noul is not a probability in [0, 1]."""


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
            "TypeSafe Jev returned malformed JSON."
        ) from exc

    try:
        probability = payload["answers"]["match"]["noul"]
    except (KeyError, TypeError) as exc:
        raise JevAnswerMissingError(
            "TypeSafe Jev response is missing answers.match.noul."
        ) from exc

    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise JevInvalidProbabilityError(
            "TypeSafe Jev answers.match.noul must be a finite number in [0, 1]."
        )

    return float(probability)


class TypeSafeJevAdapter(Classifier):
    """Score a structured image context against a query using TypeSafe Jev."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Opener | None = None,
    ) -> None:
        api_token = os.getenv("TYPESAFE_API_KEY")
        if not api_token:
            raise JevConfigurationError(
                "TYPESAFE_API_KEY is required to use TypeSafeJevAdapter."
            )

        self._api_token = api_token
        self._timeout = timeout
        self._opener = opener or urlopen

    def score(self, context: Context, query: str) -> Judgement:
        request_payload = {
            "state": context.model_dump(mode="json"),
            "questions": {
                "match": {
                    "type": "noul",
                    "instructions": (
                        "Does the provided image context match the target concept: "
                        f"{query}?"
                    ),
                }
            },
        }
        request = Request(
            JEV_ENDPOINT,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_token}",
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
                f"TypeSafe Jev returned HTTP status {exc.code}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise JevNetworkError("Could not reach the TypeSafe Jev API.") from exc

        if not 200 <= status < 300:
            raise JevHTTPError(f"TypeSafe Jev returned HTTP status {status}.")

        return Judgement(score=_parse_probability(response_body))
