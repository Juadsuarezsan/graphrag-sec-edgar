"""Download 10-K, Exhibit 21 and DEF 14A filings for the top-100 S&P 500 companies.

Source: SEC EDGAR (public domain). Rate limit: 10 requests/second and a
descriptive ``User-Agent`` are mandatory (https://www.sec.gov/os/accessing-edgar-data).

Usage:
    python scripts/download_data.py --years 3 --out data/raw [--limit 5]

Writes:
    data/raw/<TICKER>/<accession>/<primary_document>.htm  (10-K)
    data/raw/<TICKER>/<accession>/ex21*.htm               (Exhibit 21, when present)
    data/raw/<TICKER>/<accession>/def14a.htm              (proxy statement)
    data/MANIFEST.txt  (sha256 of every downloaded file)

The CIK list is committed in ``data/sp500_top100.csv``. This network blocks
``sec.gov``; the script is exercised through ``respx`` mocks in the test suite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CIKS = ROOT / "data" / "sp500_top100.csv"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/index.json"
MIN_INTERVAL_S = 0.11  # < 10 req/s


class TransientHTTP(RuntimeError):
    """429 / 5xx from EDGAR — retried with backoff."""


@dataclass(frozen=True)
class Company:
    """One row of ``data/sp500_top100.csv``."""

    rank: int
    ticker: str
    cik: str
    name: str


@dataclass
class Downloaded:
    """One file written to disk."""

    ticker: str
    form: str
    accession: str
    filing_date: str
    path: Path
    sha256: str
    bytes: int


def load_companies(path: Path = DEFAULT_CIKS, limit: int | None = None) -> list[Company]:
    """Read the committed CIK list."""
    rows: list[Company] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(Company(int(row["rank"]), row["ticker"], row["cik"].zfill(10), row["name"]))
    rows.sort(key=lambda c: c.rank)
    return rows[:limit] if limit else rows


class EdgarClient:
    """Polite EDGAR HTTP client: User-Agent, 10 req/s throttle, timeouts, retries."""

    def __init__(
        self, user_agent: str, timeout_s: float = 30.0, client: httpx.Client | None = None
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email")
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout_s,
        )
        self._last = 0.0

    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, TransientHTTP)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    def get(self, url: str) -> httpx.Response:
        """GET with throttle and retry; raises for 4xx other than 429."""
        self._throttle()
        resp = self._client.get(url)
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("EDGAR {} on {}", resp.status_code, url)
            raise TransientHTTP(f"{resp.status_code} {url}")
        resp.raise_for_status()
        return resp

    def submissions(self, cik: str) -> dict[str, Any]:
        """``data.sec.gov/submissions/CIK##########.json``."""
        data: dict[str, Any] = self.get(SUBMISSIONS_URL.format(cik=cik)).json()
        return data

    def filing_index(self, cik: str, accession: str) -> list[str]:
        """Files inside a filing folder (``index.json`` → names)."""
        url = INDEX_URL.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""))
        data = self.get(url).json()
        return [item["name"] for item in data.get("directory", {}).get("item", [])]

    def document(self, cik: str, accession: str, doc: str) -> bytes:
        """Raw bytes of one filing document."""
        return self.get(
            ARCHIVE_URL.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""), doc=doc)
        ).content


def select_filings(
    submissions: dict[str, Any], forms: tuple[str, ...], since: date
) -> list[dict[str, str]]:
    """Recent filings of the given forms filed on/after ``since``."""
    recent = submissions.get("filings", {}).get("recent", {})
    out: list[dict[str, str]] = []
    for form, acc, fdate, doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", []),
        strict=False,
    ):
        if form in forms and date.fromisoformat(fdate) >= since:
            out.append(
                {"form": form, "accession": acc, "filing_date": fdate, "primary_document": doc}
            )
    return out


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256."""
    return hashlib.sha256(data).hexdigest()


def write_manifest(path: Path, rows: list[Downloaded], root: Path) -> None:
    """Write ``MANIFEST.txt`` (``sha256  relative/path  bytes  form  accession  filing_date``)."""
    lines = [
        "# sha256  path  bytes  form  accession  filing_date",
        f"# generated {date.today().isoformat()} by scripts/download_data.py",
    ]
    for r in sorted(rows, key=lambda r: str(r.path)):
        rel = r.path.relative_to(root) if r.path.is_relative_to(root) else r.path
        lines.append(
            f"{r.sha256}  {rel.as_posix()}  {r.bytes}  {r.form}  {r.accession}  {r.filing_date}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_company(
    client: EdgarClient, company: Company, out: Path, since: date, want_ex21: bool = True
) -> list[Downloaded]:
    """Download every wanted filing for one company."""
    subs = client.submissions(company.cik)
    filings = select_filings(subs, ("10-K", "DEF 14A"), since)
    results: list[Downloaded] = []
    for f in filings:
        folder = out / company.ticker / f["accession"]
        folder.mkdir(parents=True, exist_ok=True)
        docs = [f["primary_document"]]
        if f["form"] == "10-K" and want_ex21:
            names = client.filing_index(company.cik, f["accession"])
            docs.extend(n for n in names if "ex21" in n.lower() or "ex-21" in n.lower())
        for doc in docs:
            data = client.document(company.cik, f["accession"], doc)
            target = folder / (Path(doc).name if f["form"] == "10-K" else "def14a.htm")
            target.write_bytes(data)
            results.append(
                Downloaded(
                    company.ticker,
                    f["form"],
                    f["accession"],
                    f["filing_date"],
                    target,
                    sha256_bytes(data),
                    len(data),
                )
            )
            logger.info(
                "{} {} {} -> {} ({} bytes)",
                company.ticker,
                f["form"],
                f["accession"],
                target.name,
                len(data),
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--ciks", default=str(DEFAULT_CIKS))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first N companies")
    parser.add_argument("--user-agent", default="graphrag-sec-edgar juadsuarezsan@unal.edu.co")
    parser.add_argument("--no-ex21", action="store_true")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    out = Path(args.out)
    since = date(date.today().year - args.years, 1, 1)
    client = EdgarClient(args.user_agent)
    rows: list[Downloaded] = []
    for company in load_companies(Path(args.ciks), args.limit):
        try:
            rows.extend(download_company(client, company, out, since, want_ex21=not args.no_ex21))
        except (httpx.HTTPStatusError, TransientHTTP, httpx.TransportError) as exc:
            logger.error("{}: giving up ({})", company.ticker, exc)
    write_manifest(out.parent / "MANIFEST.txt", rows, out.parent)
    logger.info("downloaded {} files; manifest at {}", len(rows), out.parent / "MANIFEST.txt")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
