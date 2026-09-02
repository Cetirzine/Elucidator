from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from elucidator.contracts import SourceClass, TimedNewsRecord, assert_available_at


def make_record(available_at: datetime) -> TimedNewsRecord:
    published_at = datetime(2026, 7, 31, 8, 0, tzinfo=UTC)
    return TimedNewsRecord(
        record_id="news-1",
        source_id="source-1",
        source_class=SourceClass.LICENSED_MEDIA,
        title="Example",
        body="Example body",
        published_at=published_at,
        provider_first_seen_at=published_at + timedelta(minutes=1),
        ingested_at=published_at + timedelta(minutes=2),
        available_at=available_at,
        content_hash="sha256:example",
        data_version="fixture-v1",
    )


def test_rejects_record_unavailable_at_forecast_origin() -> None:
    origin = datetime(2026, 7, 31, 8, 5, tzinfo=UTC)
    record = make_record(origin + timedelta(seconds=1))
    with pytest.raises(ValueError, match="after forecast origin"):
        assert_available_at(record, origin)


def test_accepts_record_available_at_forecast_origin() -> None:
    origin = datetime(2026, 7, 31, 8, 5, tzinfo=UTC)
    assert_available_at(make_record(origin), origin)


def test_rejects_naive_timestamps() -> None:
    aware = datetime(2026, 7, 31, 8, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        TimedNewsRecord(
            record_id="news-2",
            source_id="source-1",
            source_class=SourceClass.OPEN_WEB,
            title="Example",
            body="Example body",
            published_at=datetime(2026, 7, 31, 8, 0),
            provider_first_seen_at=aware,
            ingested_at=aware,
            available_at=aware,
            content_hash="sha256:example",
            data_version="fixture-v1",
        )
