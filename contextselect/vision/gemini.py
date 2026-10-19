"""Gemini 2.5 Flash context generator."""

import os
from pathlib import Path
from typing import Protocol, TypeAlias

from google import genai
from google.genai import types
from pydantic import ValidationError

from contextselect.schema import Context
from contextselect.vision.base import ContextGenerator

MODEL_NAME = "gemini-2.5-flash"
PROMPT_PATH = Path(__file__).parent / "prompts" / "context_v1.txt"
ImageInput: TypeAlias = bytes | bytearray | str | Path

_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


class GeminiContextError(RuntimeError):
    """Base exception for Gemini context generation failures."""


class GeminiConfigurationError(GeminiContextError):
    """Raised when Gemini configuration is missing."""


class ImageInputError(GeminiContextError):
    """Raised when the supplied image is unreadable or unsupported."""


class _MalformedResponseError(ValueError):
    """Internal signal for responses that cannot be validated as Context."""


class _TextResponse(Protocol):
    text: str | None


def _image_data(image: ImageInput) -> tuple[bytes, str]:
    if isinstance(image, (str, Path)):
        try:
            data = Path(image).read_bytes()
        except OSError as exc:
            raise ImageInputError(f"Could not read image file: {image}") from exc
    elif isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    else:
        raise ImageInputError("Image must be bytes or a path to a JPEG, PNG, or WebP file.")

    if not data:
        raise ImageInputError("Image data is empty.")

    for signature, mime_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return data, mime_type

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, "image/webp"

    raise ImageInputError("Image data must be a JPEG, PNG, or WebP image.")


def _context_from_response(response: _TextResponse) -> Context:
    try:
        response_text = response.text
    except ValueError as exc:
        raise _MalformedResponseError("Gemini response did not contain readable text.") from exc

    if not isinstance(response_text, str) or not response_text.strip():
        raise _MalformedResponseError("Gemini response was empty.")

    try:
        return Context.model_validate_json(response_text)
    except ValidationError as exc:
        raise _MalformedResponseError(
            "Gemini response was not valid JSON matching the Context schema."
        ) from exc


class GeminiContextGenerator(ContextGenerator[ImageInput]):
    """Generate query-agnostic structured context with Gemini 2.5 Flash."""

    def __init__(self, client: genai.Client | None = None) -> None:
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise GeminiConfigurationError(
                    "GEMINI_API_KEY is required to use GeminiContextGenerator."
                )
            client = genai.Client(api_key=api_key)

        self._client = client
        self._prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Context,
            temperature=0,
        )

    def generate(self, image: ImageInput) -> Context:
        image_bytes, mime_type = _image_data(image)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        prompt = self._prompt

        for attempt in range(2):
            response = self._client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image_part],
                config=self._config,
            )
            try:
                return _context_from_response(response)
            except _MalformedResponseError as exc:
                if attempt == 1:
                    raise GeminiContextError(
                        "Gemini returned malformed context after one retry."
                    ) from exc
                prompt = (
                    f"{self._prompt}\n\n"
                    "Your previous response did not match the required JSON schema. "
                    "Correct it and return only valid JSON matching the schema."
                )

        raise AssertionError("The bounded generation retry loop must return or raise.")
