import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from openai import OpenAI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from elucidator.contracts import ExtractedNewsFactor, ExtractionResult, TimedNewsRecord
from elucidator.settings import Settings

PROMPT_VERSION = "news-factor-v1.0.0"

SYSTEM_PROMPT = """You extract point-in-time financial news facts for quantitative research.
Do not give investment advice or predict a price. Use only the supplied article, distinguish facts
from claims, and return one JSON object. Required JSON keys: schema_version, symbols, event_type,
polarity, materiality, uncertainty, relevance, horizon, facts, claims, confidence. Numeric values
must be in the documented ranges; schema_version must be news-factor-v1.
"""


class NewsFactorExtractor:
    """Provider-neutral OpenAI-compatible adapter configured for DeepSeek by default."""

    def __init__(self, settings: Settings) -> None:
        if settings.llm_api_key is None:
            raise ValueError("LLM_API_KEY (or DEEPSEEK_API_KEY) is required")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    def extract(self, record: TimedNewsRecord) -> ExtractionResult:
        payload = {
            "record_id": record.record_id,
            "source_class": record.source_class.value,
            "published_at": record.published_at.isoformat(),
            "title": record.title,
            "body": record.body,
        }
        user_text = "Return JSON for this article:\n" + json.dumps(payload, ensure_ascii=False)
        digest = hashlib.sha256(user_text.encode("utf-8")).hexdigest()

        response: Any = None
        attempts = max(1, self.settings.llm_max_retries + 1)
        for attempt in Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
            reraise=True,
        ):
            with attempt:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty JSON content")
                parsed = json.loads(content)
                factor = ExtractedNewsFactor.model_validate(parsed)

        return ExtractionResult(
            factor=factor,
            input_sha256=digest,
            prompt_version=PROMPT_VERSION,
            requested_model=self.settings.llm_model,
            response_model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
            created_at=datetime.now(UTC),
            raw_json=content,
        )

