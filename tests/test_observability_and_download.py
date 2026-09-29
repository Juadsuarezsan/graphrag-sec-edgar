"""Observability (cost, metrics, LangSmith exporter) and the EDGAR download script."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from scripts.download_data import (
    ARCHIVE_URL,
    INDEX_URL,
    SUBMISSIONS_URL,
    EdgarClient,
    TransientHTTP,
    download_company,
    load_companies,
    select_filings,
    write_manifest,
)
from src.config import Settings, get_settings
from src.observability import (
    LangSmithTracer,
    RequestMetrics,
    current_trace_id,
    estimate_cost_usd,
    new_trace_id,
)


def test_cost_and_metrics() -> None:
    assert estimate_cost_usd("claude-sonnet-4-5-20250929", 1_000_000, 1_000_000) == 18.0
    assert estimate_cost_usd("unknown-model", 10, 10) == 0.0
    tid = new_trace_id()
    assert current_trace_id() == tid and len(tid) == 16
    m = RequestMetrics(name="t")
    m.add_usage("claude-sonnet-4-5-20250929", 1000, 500)
    m.finish()
    assert m.llm_calls == 1 and m.cost_usd == pytest.approx(0.0105) and m.latency_ms >= 0
    assert "started_at" not in m.as_dict() and m.as_dict()["trace_id"] == tid


@respx.mock
def test_langsmith_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    assert LangSmithTracer(Settings()).export(RequestMetrics(), {}, {}) is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
    get_settings.cache_clear()
    settings = get_settings()
    route = respx.post("https://api.smith.langchain.com/runs").mock(
        return_value=httpx.Response(202)
    )
    tracer = LangSmithTracer(settings, client=httpx.Client())
    assert (
        tracer.enabled
        and tracer.export(RequestMetrics(name="query"), {"q": "x"}, {"a": "y"}) is True
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["session_name"] == "graphrag-sec-edgar" and sent["inputs"] == {"q": "x"}
    assert route.calls[0].request.headers["x-api-key"] == "ls-key"
    route.mock(return_value=httpx.Response(500))
    tracer._post.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    assert tracer.export(RequestMetrics(), {}, {}) is False


def _submissions() -> dict[str, object]:
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "DEF 14A", "10-K"],
                "accessionNumber": [
                    "0000320193-25-000079",
                    "0000320193-25-000010",
                    "0000320193-25-000005",
                    "0000320193-20-000001",
                ],
                "filingDate": ["2025-10-31", "2025-05-01", "2025-01-10", "2020-10-30"],
                "primaryDocument": ["aapl-20250927.htm", "8k.htm", "proxy.htm", "old.htm"],
            }
        }
    }


def test_load_companies_and_select_filings() -> None:
    companies = load_companies(limit=3)
    assert [c.ticker for c in companies] == ["NVDA", "MSFT", "AAPL"] and companies[
        2
    ].cik == "0000320193"
    assert len(load_companies()) == 100
    picked = select_filings(_submissions(), ("10-K", "DEF 14A"), date(2023, 1, 1))
    assert [p["form"] for p in picked] == ["10-K", "DEF 14A"]


@respx.mock
def test_download_company_writes_files_and_manifest(tmp_path: Path) -> None:
    cik = "0000320193"
    respx.get(SUBMISSIONS_URL.format(cik=cik)).mock(
        return_value=httpx.Response(200, json=_submissions())
    )
    respx.get(INDEX_URL.format(cik_int=320193, acc_nodash="000032019325000079")).mock(
        return_value=httpx.Response(
            200,
            json={
                "directory": {
                    "item": [{"name": "aapl-20250927.htm"}, {"name": "aapl-20250927_ex21.htm"}]
                }
            },
        )
    )
    respx.get(
        ARCHIVE_URL.format(cik_int=320193, acc_nodash="000032019325000079", doc="aapl-20250927.htm")
    ).mock(return_value=httpx.Response(200, content=b"<html>10-K</html>"))
    respx.get(
        ARCHIVE_URL.format(
            cik_int=320193, acc_nodash="000032019325000079", doc="aapl-20250927_ex21.htm"
        )
    ).mock(return_value=httpx.Response(200, content=b"<html>ex21</html>"))
    respx.get(
        ARCHIVE_URL.format(cik_int=320193, acc_nodash="000032019325000005", doc="proxy.htm")
    ).mock(return_value=httpx.Response(200, content=b"<html>proxy</html>"))
    client = EdgarClient(
        "tests test@example.com",
        client=httpx.Client(headers={"User-Agent": "tests test@example.com"}),
    )
    company = load_companies(limit=3)[2]
    rows = download_company(client, company, tmp_path / "raw", date(2023, 1, 1))
    assert sorted(r.path.name for r in rows) == [
        "aapl-20250927.htm",
        "aapl-20250927_ex21.htm",
        "def14a.htm",
    ]
    assert all(len(r.sha256) == 64 for r in rows)
    write_manifest(tmp_path / "MANIFEST.txt", rows, tmp_path)
    manifest = (tmp_path / "MANIFEST.txt").read_text()
    assert "raw/AAPL/0000320193-25-000079/aapl-20250927.htm" in manifest and "DEF 14A" in manifest


@respx.mock
def test_edgar_client_retries_and_user_agent() -> None:
    with pytest.raises(ValueError):
        EdgarClient("no-email-here")
    url = SUBMISSIONS_URL.format(cik="0000000001")
    route = respx.get(url).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": 1})]
    )
    client = EdgarClient("t t@e.com", client=httpx.Client())
    client.get.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    assert client.submissions("0000000001") == {"ok": 1} and route.call_count == 2
    respx.get(url).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        client.submissions("0000000001")
    respx.get(url).mock(return_value=httpx.Response(429))
    with pytest.raises(TransientHTTP):
        client.submissions("0000000001")
