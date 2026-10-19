from pathlib import Path
from unittest.mock import Mock

import pytest

from contextselect.schema import Context
from contextselect.vision.gemini import (
    MODEL_NAME,
    GeminiConfigurationError,
    GeminiContextError,
    GeminiContextGenerator,
    ImageInputError,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nmock-image"
VALID_CONTEXT_JSON = """
{
  "subject": "ceramic mug",
  "category": "drinkware",
  "appearance": "A small white vessel with a curved handle.",
  "distinctive_features": ["curved handle"],
  "function": "Used for holding a drink (inference).",
  "state_condition": "Intact; no visible damage.",
  "action_context": "No action is apparent.",
  "environment": "On a wooden table.",
  "visible_text": [],
  "related_concepts": ["cup"],
  "uncertainty": ""
}
"""


def make_generator(response_texts: list[str]) -> tuple[GeminiContextGenerator, Mock]:
    client = Mock()
    client.models.generate_content.side_effect = [
        Mock(text=response_text) for response_text in response_texts
    ]
    return GeminiContextGenerator(client=client), client


def test_generates_and_validates_structured_context() -> None:
    generator, client = make_generator([VALID_CONTEXT_JSON])

    context = generator.generate(PNG_BYTES)

    assert isinstance(context, Context)
    assert context.subject == "ceramic mug"
    request = client.models.generate_content.call_args.kwargs
    assert request["model"] == MODEL_NAME
    assert request["config"].response_mime_type == "application/json"
    assert request["config"].temperature == 0
    response_schema = request["config"].response_schema
    assert response_schema["type"] == "OBJECT"
    assert response_schema["required"] == list(Context.model_fields)

    def assert_no_unsupported_additional_properties(schema: object) -> None:
        if isinstance(schema, dict):
            assert "additionalProperties" not in schema
            assert "additional_properties" not in schema
            for value in schema.values():
                assert_no_unsupported_additional_properties(value)
        elif isinstance(schema, list):
            for value in schema:
                assert_no_unsupported_additional_properties(value)

    assert_no_unsupported_additional_properties(response_schema)
    assert "query-agnostic" in request["contents"][0]
    assert request["contents"][1].inline_data.mime_type == "image/png"


def test_uses_validated_parsed_context_when_sdk_provides_it() -> None:
    generator, client = make_generator(["not used"])
    parsed_context = Context.model_validate_json(VALID_CONTEXT_JSON)
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value = Mock(
        parsed=parsed_context,
        text=None,
    )

    result = generator.generate(PNG_BYTES)

    assert result is parsed_context
    client.models.generate_content.assert_called_once()


def test_parsed_dict_is_still_validated_against_strict_context_schema() -> None:
    generator, client = make_generator(["not used"])
    parsed_context = Context.model_validate_json(VALID_CONTEXT_JSON).model_dump()
    parsed_context["unexpected"] = "extra field"
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value = Mock(
        parsed=parsed_context,
        text=None,
    )

    with pytest.raises(GeminiContextError, match="after one retry"):
        generator.generate(PNG_BYTES)

    assert client.models.generate_content.call_count == 2


def test_retries_malformed_response_once_then_succeeds() -> None:
    generator, client = make_generator(["{invalid json", VALID_CONTEXT_JSON])

    context = generator.generate(PNG_BYTES)

    assert context.subject == "ceramic mug"
    assert client.models.generate_content.call_count == 2
    retry_prompt = client.models.generate_content.call_args.kwargs["contents"][0]
    assert "previous response did not match" in retry_prompt


def test_raises_custom_error_after_retry_fails() -> None:
    generator, client = make_generator(["{}", "{still invalid"])

    with pytest.raises(GeminiContextError, match="after one retry"):
        generator.generate(PNG_BYTES)

    assert client.models.generate_content.call_count == 2


def test_requires_api_key_when_client_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiContextGenerator()


@pytest.mark.parametrize(
    ("image", "expected_mime_type", "expected_bytes"),
    [
        (PNG_BYTES, "image/png", PNG_BYTES),
        (bytearray(PNG_BYTES), "image/png", PNG_BYTES),
    ],
)
def test_accepts_image_bytes(
    image: bytes | bytearray, expected_mime_type: str, expected_bytes: bytes
) -> None:
    generator, client = make_generator([VALID_CONTEXT_JSON])

    generator.generate(image)

    part = client.models.generate_content.call_args.kwargs["contents"][1]
    assert part.inline_data.mime_type == expected_mime_type
    assert part.inline_data.data == expected_bytes


def test_reads_image_from_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(PNG_BYTES)
    generator, client = make_generator([VALID_CONTEXT_JSON])

    generator.generate(image_path)

    part = client.models.generate_content.call_args.kwargs["contents"][1]
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == PNG_BYTES


def test_rejects_unsupported_image_input() -> None:
    generator, client = make_generator([VALID_CONTEXT_JSON])

    with pytest.raises(ImageInputError, match="JPEG, PNG, or WebP"):
        generator.generate(b"not an image")

    client.models.generate_content.assert_not_called()
