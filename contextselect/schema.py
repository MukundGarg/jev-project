"""Validated data models shared by ContextSelect components."""

from pydantic import BaseModel, ConfigDict, Field


class Context(BaseModel):
    """Query-agnostic semantic description of one image."""

    model_config = ConfigDict(extra="forbid", strict=True)

    subject: str
    category: str
    appearance: str
    distinctive_features: list[str]
    function: str
    state_condition: str
    action_context: str
    environment: str
    visible_text: list[str]
    related_concepts: list[str]
    uncertainty: str


class Judgement(BaseModel):
    """Jev's relevance probability for one context and query."""

    score: float = Field(ge=0.0, le=1.0)
