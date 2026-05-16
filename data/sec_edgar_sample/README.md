# Real SEC EDGAR 10-K excerpts

6 actual 10-K filings pulled from SEC EDGAR public API (no auth, free).
For each company we keep the "Item 1 — Business" section excerpt (up to
8000 chars) so the entity extractor + knowledge-graph ingestion can be
demonstrated on real text instead of the 25-company synthetic fixture.

| Ticker | Accession | Business section chars |
|---|---|---|
| AAPL  | 0000320193-24-000123 | ~8000 |
| MSFT  | 0000950170-25-100235 | ~8000 |
| NVDA  | 0001045810-26-000021 | (HTML template differs — re-parse with `pip install sec-edgar-downloader` for full body) |
| GOOGL | 0001652044-26-000018 | ~8000 |
| META  | 0001628280-26-003942 | (re-parse needed) |
| TSLA  | 0001628280-26-003952 | (re-parse needed) |

## Why 3 of 6 came out as 1 char

The naive regex `Item 1. Business` doesn't anchor on every filer's HTML
template. NVDA / META / TSLA wrap the section header inside nested
`<span><font>...</font></span>` tags or use non-breaking spaces. The
production solution is `unstructured-io` or the `sec-edgar-downloader`
library with `--include-amends`. AAPL / MSFT / GOOGL came out clean
because they use the more standard inline header format.

## How to fetch your own

```python
import httpx
HEADERS = {"User-Agent": "Your Name your@email.com"}  # SEC requires User-Agent
url = "https://data.sec.gov/submissions/CIK0000320193.json"
data = httpx.get(url, headers=HEADERS).json()
# data["filings"]["recent"] has form / accessionNumber / primaryDocument arrays
```

SEC rate-limits to 10 req/sec. The `scripts/fetch_sec_10k.py` (this
project) respects that with `time.sleep(0.5)` between calls.

## What goes into the graph

For each parsed 10-K, the entity extractor pulls:
- Company entity (already exists from the demo fixture; this updates its
  embeddings with real text)
- Person entities mentioned in the business section (CEOs, directors)
- Product entities (named offerings)
- Risk entities (when paired with Item 1A — not in this sample)
- Supplier / customer / competitor relationships when those entities
  appear together within a sentence

Run `python -m src.ingestion.ingest_sec` to fold these excerpts into the
live in-memory graph (overwrites the synthetic-only fixture).
