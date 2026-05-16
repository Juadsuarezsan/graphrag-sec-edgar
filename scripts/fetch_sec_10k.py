"""Fetch real 10-K business sections from SEC EDGAR.

Respects SEC's 10 req/sec rate limit. Writes JSON files to
data/sec_edgar_sample/. Free, no API key needed — only a User-Agent header.

Run: python scripts/fetch_sec_10k.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

HEADERS = {"User-Agent": "Juadsuarezsan portfolio juadsuarezsan@unal.edu.co"}
OUT = Path(__file__).parent.parent / "data" / "sec_edgar_sample"

COMPANIES = [
    ("AAPL",  "0000320193", "Apple Inc."),
    ("MSFT",  "0000789019", "Microsoft Corp."),
    ("NVDA",  "0001045810", "NVIDIA Corp."),
    ("GOOGL", "0001652044", "Alphabet Inc."),
    ("META",  "0001326801", "Meta Platforms"),
    ("TSLA",  "0001318605", "Tesla Inc."),
]


def _latest_10k(cik: str) -> tuple[str, str] | None:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    with httpx.Client(headers=HEADERS, timeout=15) as c:
        data = c.get(url).raise_for_status().json()
    recent = data["filings"]["recent"]
    for form, acc, doc in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form == "10-K":
            return (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}",
                acc,
            )
    return None


def _strip(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&\w+;", " ", text)
    return re.sub(r"\s+", " ", text)


def _business(text: str, max_chars: int = 8000) -> str:
    m = re.search(r"Item\s*1[\.\s]+Business", text, re.IGNORECASE)
    if not m:
        return text[:max_chars]
    start = m.end()
    next_section = re.search(r"Item\s*1A[\.\s]+Risk\s+Factors", text[start:], re.IGNORECASE)
    end = start + (next_section.start() if next_section else max_chars)
    return text[start : min(len(text), end)].strip()[:max_chars]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []
    for ticker, cik, name in COMPANIES:
        print(f"[{ticker}] fetching ...")
        info = _latest_10k(cik)
        if not info:
            continue
        url, accession = info
        time.sleep(0.5)  # 10 req/sec → 0.5s between requests
        with httpx.Client(headers=HEADERS, timeout=30) as c:
            html = c.get(url).text
        body = _business(_strip(html))
        (OUT / f"{ticker}_10K.json").write_text(
            json.dumps({
                "ticker": ticker, "cik": cik, "company_name": name,
                "accession": accession, "source_url": url,
                "business_section_excerpt": body,
                "business_section_chars": len(body),
            }, indent=2),
            encoding="utf-8",
        )
        catalog.append({"ticker": ticker, "company_name": name,
                         "accession": accession, "excerpt_chars": len(body)})
        print(f"  wrote {ticker}_10K.json  ({len(body)} chars)")
        time.sleep(0.5)
    (OUT / "_index.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"\nDone. {len(catalog)} excerpts.")


if __name__ == "__main__":
    main()
