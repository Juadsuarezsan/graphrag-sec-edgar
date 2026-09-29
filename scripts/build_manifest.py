"""Regenerate ``data/MANIFEST.txt`` with the SHA-256 of every committed data file.

Run: python scripts/build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    *sorted((ROOT / "data" / "sec_edgar_sample").glob("*.json")),
    ROOT / "data" / "sp500_top100.csv",
    ROOT / "eval" / "gold.jsonl",
    ROOT / "eval" / "extraction_gold.jsonl",
]


def main() -> None:
    """Write the manifest."""
    lines = [
        "# sha256  path  bytes  form  accession  filing_date",
        f"# generated {datetime.now(UTC).date().isoformat()} by scripts/build_manifest.py",
    ]
    for path in FILES:
        data = path.read_bytes()
        form = accession = fdate = "-"
        if path.suffix == ".json" and path.name.endswith("_10K.json"):
            raw = json.loads(data)
            form, accession = "10-K", raw.get("accession", "-")
        rel = path.relative_to(ROOT).as_posix()
        lines.append(
            f"{hashlib.sha256(data).hexdigest()}  {rel}  {len(data)}  {form}  {accession}  {fdate}"
        )
    out = ROOT / "data" / "MANIFEST.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(FILES)} entries to {out}")


if __name__ == "__main__":
    main()
