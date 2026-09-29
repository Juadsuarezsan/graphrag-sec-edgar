"""Observability primitives: trace ids, per-request metrics, cost, LangSmith.

The module is intentionally dependency-free (no LangSmith SDK): traces are
posted with ``httpx`` to the LangSmith REST API only when
``LANGSMITH_API_KEY`` is configured, so the wiring is real but inert by
default.
"""

from __future__ import annotations

import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings

#: USD per million tokens (input, output) for the pinned production model.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "voyage-3": (0.06, 0.0),
}

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    """Create and bind a new trace id for the current context."""
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def current_trace_id() -> str:
    """Return the trace id bound to the current context (``-`` if none)."""
    return _trace_id.get()


def configure_logging(level: str | None = None) -> None:
    """Configure loguru with a structured, trace-aware format."""
    lvl = level or get_settings().log_level
    logger.remove()
    logger.configure(patcher=_inject_trace_id)
    logger.add(
        sys.stderr,
        level=lvl,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "trace={extra[trace_id]} | {name}:{function} | {message}"
        ),
    )


def _inject_trace_id(record: Any) -> None:
    """loguru patcher: make ``trace_id`` available to every format string."""
    record["extra"].setdefault("trace_id", current_trace_id())


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate USD cost from token counts using the pricing table.

    Args:
        model: Model id; unknown models cost 0 and are logged once.
        tokens_in: Prompt tokens.
        tokens_out: Completion tokens.

    Returns:
        Cost in USD rounded to 6 decimals.
    """
    if model not in MODEL_PRICING_USD_PER_MTOK:
        logger.warning("no pricing for model {}; cost reported as 0", model)
        return 0.0
    price_in, price_out = MODEL_PRICING_USD_PER_MTOK[model]
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)


@dataclass
class RequestMetrics:
    """Metrics accumulated during one request or one pipeline step.

    Attributes:
        trace_id: Correlation id.
        name: Logical operation name (``query``, ``extract`` ...).
        started_at: ``perf_counter`` at start.
        latency_ms: Filled by :meth:`finish`.
        tokens_in: LLM prompt tokens consumed.
        tokens_out: LLM completion tokens consumed.
        cost_usd: Estimated USD cost.
        llm_calls: Number of LLM calls.
        extra: Free-form structured fields.
    """

    trace_id: str = field(default_factory=current_trace_id)
    name: str = "request"
    started_at: float = field(default_factory=time.perf_counter)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def add_usage(self, model: str, tokens_in: int, tokens_out: int) -> None:
        """Accumulate one LLM call's usage and its cost."""
        self.llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd = round(self.cost_usd + estimate_cost_usd(model, tokens_in, tokens_out), 6)

    def finish(self) -> RequestMetrics:
        """Freeze latency and emit one structured log line."""
        self.latency_ms = int((time.perf_counter() - self.started_at) * 1000)
        logger.bind(trace_id=self.trace_id).info(
            "{} done latency_ms={} tokens_in={} tokens_out={} cost_usd={} llm_calls={} {}",
            self.name,
            self.latency_ms,
            self.tokens_in,
            self.tokens_out,
            self.cost_usd,
            self.llm_calls,
            self.extra,
        )
        return self

    def as_dict(self) -> dict[str, Any]:
        """Serializable view (drops the monotonic start time)."""
        d = asdict(self)
        d.pop("started_at", None)
        return d


class LangSmithTracer:
    """Minimal LangSmith run exporter over the public REST API.

    Active only when ``LANGSMITH_API_KEY`` is set. Failures are logged and
    never propagate to the request path.
    """

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self._settings = settings or get_settings()
        self._client = client

    @property
    def enabled(self) -> bool:
        """Whether a key is configured."""
        return bool(self._settings.langsmith_api_key)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2),
        reraise=True,
    )
    def _post(self, payload: dict[str, Any]) -> None:
        url = f"{self._settings.langsmith_endpoint.rstrip('/')}/runs"
        headers = {"x-api-key": self._settings.langsmith_api_key or ""}
        if self._client is not None:
            resp = self._client.post(url, json=payload, headers=headers)
        else:
            with httpx.Client(timeout=self._settings.request_timeout_s) as client:
                resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    def export(
        self, metrics: RequestMetrics, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> bool:
        """Export one run. Returns ``True`` only when actually sent."""
        if not self.enabled:
            return False
        payload = {
            "id": str(uuid.uuid4()),
            "name": metrics.name,
            "run_type": "chain",
            "session_name": self._settings.langsmith_project,
            "start_time": datetime.now(UTC).isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "extra": {"metrics": metrics.as_dict()},
        }
        try:
            self._post(payload)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            logger.warning("langsmith export failed: {}", exc)
            return False
        return True
