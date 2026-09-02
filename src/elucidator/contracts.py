from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, model_validator


class SourceClass(StrEnum):
    EXCHANGE = "exchange"
    REGULATOR = "regulator"
    COMPANY = "company"
    LICENSED_MEDIA = "licensed_media"
    OPEN_WEB = "open_web"
    SOCIAL = "social"


class TimedNewsRecord(BaseModel):
    """Raw-news timing contract used to prevent point-in-time leakage."""

    record_id: str
    source_id: str
    source_class: SourceClass
    title: str
    body: str
    event_time: AwareDatetime | None = None
    published_at: AwareDatetime
    provider_first_seen_at: AwareDatetime
    ingested_at: AwareDatetime
    available_at: AwareDatetime
    content_hash: str
    data_version: str

    @model_validator(mode="after")
    def validate_timing(self) -> "TimedNewsRecord":
        if self.provider_first_seen_at < self.published_at:
            raise ValueError("provider_first_seen_at must not precede published_at")
        if self.available_at < self.published_at:
            raise ValueError("available_at must not precede published_at")
        return self


class ExtractedNewsFactor(BaseModel):
    """Versioned, locally validated factor schema returned by an LLM."""

    schema_version: Literal["news-factor-v1"] = "news-factor-v1"
    symbols: list[str] = Field(default_factory=list)
    event_type: str
    polarity: float = Field(ge=-1.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    horizon: Literal["intraday", "days", "weeks", "months", "unknown"]
    facts: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    factor: ExtractedNewsFactor
    input_sha256: str
    prompt_version: str
    requested_model: str
    response_model: str | None = None
    system_fingerprint: str | None = None
    created_at: datetime
    raw_json: str


def assert_available_at(record: TimedNewsRecord, forecast_origin: AwareDatetime) -> None:
    """Reject any feature that was unavailable when the simulated forecast was made."""

    if record.available_at > forecast_origin:
        raise ValueError(
            f"record {record.record_id!r} became available at {record.available_at.isoformat()}, "
            f"after forecast origin {forecast_origin.isoformat()}"
        )

