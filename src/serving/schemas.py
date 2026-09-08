"""Request/response models, plus the validation table from SPEC.md §5.

Pydantic checks that a field is a string. It does not check that the string is
sane, and this is the trust boundary — hence `check_paper`.
"""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel

from .preprocessing import MAX_INPUT_CHARS


class PredictRequest(BaseModel):
    title: str
    abstract: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    top3: list[str]
    model_version: str
    latency_ms: float

    # `model_version` collides with pydantic's protected `model_` namespace.
    # The field name is the published API contract (SPEC §4), so the namespace
    # gives way, not the name.
    model_config = {"protected_namespaces": ()}


class BatchRequest(BaseModel):
    # Length is checked in the route, not here: a Field(max_length=...) violation
    # is a 422, and SPEC §5 says an oversized batch is a 400. The cap is also a
    # setting, which a class-level annotation cannot read.
    items: list[PredictRequest]


class BatchResponse(BaseModel):
    results: list[PredictResponse]
    model_version: str
    # One figure for the whole forward pass — that is the number the batch
    # endpoint exists to make, so a per-item split would be a fiction.
    latency_ms: float

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    uptime_s: float

    model_config = {"protected_namespaces": ()}


class Metadata(BaseModel):
    """The model card, verbatim. Declared loosely on purpose: the card is the
    source of truth and a schema that pinned its keys would be a second one."""

    name: str
    version: str
    labels: list[str]
    max_input_chars: int
    max_tokens: int

    model_config = {"extra": "allow"}


def check_paper(title: str, abstract: str, index: int | None = None) -> None:
    """SPEC §5, in code. Raises 400; `index` names the offending batch item.

    Runs before join(), which truncates silently. Truncation is right inside the
    model layer and wrong at the boundary: a caller who sent 10MB should be told,
    not handed a confident answer about the first 4000 characters.
    """
    where = "" if index is None else f" at index {index}"
    for name, value in (("title", title), ("abstract", abstract)):
        if not value.strip():
            raise HTTPException(400, f"{name} is empty or whitespace-only{where}")
    total = len(title.strip()) + len(abstract.strip())
    if total > MAX_INPUT_CHARS:
        raise HTTPException(
            400,
            f"title + abstract is {total} characters{where}, "
            f"limit is {MAX_INPUT_CHARS}",
        )
