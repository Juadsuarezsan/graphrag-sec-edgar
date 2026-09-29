"""Serialize the hand-annotated extraction gold to ``eval/extraction_gold.jsonl``.

Five chunks: two synthetic structured exhibits (Exhibit 21, DEF 14A), two
synthetic Item 1 / 1A passages and one **real** passage from Alphabet's
10-K excerpt in ``data/sec_edgar_sample/GOOGL_10K.json``. Entities and
relationships were annotated by hand from the text.

Run: python scripts/build_extraction_gold.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "extraction_gold.jsonl"

APPLE = {
    "ticker": "AAPL",
    "cik": "0000320193",
    "company_name": "Apple Inc.",
    "accession": "0000320193-25-000079",
    "form_type": "10-K",
    "filing_date": "2025-10-31",
}
GOOGL = {
    "ticker": "GOOGL",
    "cik": "0001652044",
    "company_name": "Alphabet Inc.",
    "accession": "0001652044-26-000018",
    "form_type": "10-K",
    "filing_date": "2026-02-04",
}
FOCAL_APPLE = {
    "id": "apple",
    "type": "Company",
    "name": "Apple Inc.",
    "properties": {"ticker": "AAPL"},
}
FOCAL_GOOGL = {
    "id": "google",
    "type": "Company",
    "name": "Alphabet Inc.",
    "properties": {"ticker": "GOOGL"},
}

EX21_TEXT = (
    "Exhibit 21.1\n"
    "Subsidiaries of Apple Inc.\n"
    "Name | Jurisdiction of Incorporation\n"
    "Apple Operations International Limited | Ireland\n"
    "Beats Electronics, LLC | Delaware\n"
    "Braeburn Capital, Inc. | Nevada\n"
    "Apple Sales International | Ireland\n"
)
EX21_ENTITIES = [
    {
        "id": "sub_apple_operations_international_limited",
        "type": "Subsidiary",
        "name": "Apple Operations International Limited",
    },
    {"id": "sub_beats_electronics_llc", "type": "Subsidiary", "name": "Beats Electronics, LLC"},
    {"id": "sub_braeburn_capital_inc", "type": "Subsidiary", "name": "Braeburn Capital, Inc."},
    {
        "id": "sub_apple_sales_international",
        "type": "Subsidiary",
        "name": "Apple Sales International",
    },
]
EX21_RELS = [{"from_id": e["id"], "to_id": "apple", "type": "SUBSIDIARY_OF"} for e in EX21_ENTITIES]

DEF14A_TEXT = (
    "Proposal 1 - Election of Directors\n"
    "Name | Age | Director Since\n"
    "Irene Castellanos | 55 | 2020\n"
    "Anders Lindqvist | 63 | 2016\n"
    "Tim Cook | 64 | 2011\n"
    "Keiko Nakamura, 58, has served as a director since 2018.\n"
)
DEF14A_ENTITIES = [
    {"id": "irene_castellanos", "type": "Person", "name": "Irene Castellanos"},
    {"id": "anders_lindqvist", "type": "Person", "name": "Anders Lindqvist"},
    {"id": "tim_cook", "type": "Person", "name": "Tim Cook"},
    {"id": "keiko_nakamura", "type": "Person", "name": "Keiko Nakamura"},
]
DEF14A_RELS = [
    {"from_id": e["id"], "to_id": "apple", "type": "DIRECTOR_OF"} for e in DEF14A_ENTITIES
]

ITEM1_TEXT = (
    "The Company designs, manufactures and markets smartphones, personal computers and wearables. "
    "The Company's products include iPhone, Mac, iPad and Apple Watch. "
    "The markets for the Company's products are highly competitive and the Company competes with Samsung Electronics, "
    "Alphabet and Huawei. Substantially all of the Company's application processors are supplied by Taiwan Semiconductor "
    "Manufacturing Company for its iPhone and Mac product lines, and final assembly is performed by outsourcing partners. "
    "Tim Cook, our Chief Executive Officer, has led the Company since 2011."
)
ITEM1_ENTITIES = [
    {"id": "prod_iphone", "type": "Product", "name": "iPhone"},
    {"id": "prod_mac", "type": "Product", "name": "Mac"},
    {"id": "prod_ipad", "type": "Product", "name": "iPad"},
    {"id": "prod_apple_watch", "type": "Product", "name": "Apple Watch"},
    {"id": "samsung_electronics", "type": "Company", "name": "Samsung Electronics"},
    {"id": "alphabet", "type": "Company", "name": "Alphabet"},
    {"id": "huawei", "type": "Company", "name": "Huawei"},
    {
        "id": "taiwan_semiconductor_manufacturing_company",
        "type": "Company",
        "name": "Taiwan Semiconductor Manufacturing Company",
    },
    {"id": "tim_cook", "type": "Person", "name": "Tim Cook"},
]
ITEM1_RELS = [
    {"from_id": "apple", "to_id": "prod_iphone", "type": "OFFERS_PRODUCT"},
    {"from_id": "apple", "to_id": "prod_mac", "type": "OFFERS_PRODUCT"},
    {"from_id": "apple", "to_id": "prod_ipad", "type": "OFFERS_PRODUCT"},
    {"from_id": "apple", "to_id": "prod_apple_watch", "type": "OFFERS_PRODUCT"},
    {"from_id": "apple", "to_id": "samsung_electronics", "type": "COMPETES_WITH"},
    {"from_id": "apple", "to_id": "alphabet", "type": "COMPETES_WITH"},
    {"from_id": "apple", "to_id": "huawei", "type": "COMPETES_WITH"},
    {
        "from_id": "apple",
        "to_id": "taiwan_semiconductor_manufacturing_company",
        "type": "SUPPLIED_BY",
    },
    {"from_id": "tim_cook", "to_id": "apple", "type": "CEO_OF"},
]

ITEM1A_TEXT = (
    "Risk Factors. A significant portion of the Company's manufacturing is performed by outsourcing partners located in China and Taiwan, "
    "and political tensions could disrupt supply. The Company relies on a limited number of suppliers for certain components, and single source "
    "suppliers may not be able to meet demand. The Company's business could be harmed by cybersecurity incidents or a data breach. "
    "New tariffs or trade restrictions between the U.S. and China could increase the cost of the Company's products. "
    "The Company also faces antitrust investigations by competition authorities in several jurisdictions."
)
ITEM1A_ENTITIES = [
    {"id": "risk_china_exposure", "type": "Risk", "name": "China exposure"},
    {"id": "risk_supply_concentration", "type": "Risk", "name": "Supplier concentration"},
    {"id": "risk_cybersecurity", "type": "Risk", "name": "Cybersecurity incidents"},
    {"id": "risk_tariffs", "type": "Risk", "name": "Tariffs and trade restrictions"},
    {"id": "risk_antitrust", "type": "Risk", "name": "Antitrust enforcement"},
]
ITEM1A_RELS = [
    {"from_id": "apple", "to_id": e["id"], "type": "MENTIONS_RISK"} for e in ITEM1A_ENTITIES
]

# Real passage: taken verbatim from the GOOGL excerpt (Item 1, Google Services).
GOOGL_START = "Google For reporting purposes Google comprises two segments"
GOOGL_END = "with broad and growing adoption by users around the world."
GOOGL_ENTITIES = [
    {"id": f"prod_{slug}", "type": "Product", "name": name}
    for slug, name in [
        ("android", "Android"),
        ("chrome", "Chrome"),
        ("gmail", "Gmail"),
        ("google_drive", "Google Drive"),
        ("google_gemini", "Google Gemini"),
        ("google_maps", "Google Maps"),
        ("google_photos", "Google Photos"),
        ("google_play", "Google Play"),
        ("search", "Search"),
        ("youtube", "YouTube"),
    ]
]
GOOGL_RELS = [
    {"from_id": "google", "to_id": e["id"], "type": "OFFERS_PRODUCT"} for e in GOOGL_ENTITIES
]


def googl_passage() -> str:
    """Slice the real passage out of the sample JSON (fails loudly if it moved)."""
    raw = json.loads((ROOT / "data/sec_edgar_sample/GOOGL_10K.json").read_text(encoding="utf-8"))
    text = html.unescape(raw["business_section_excerpt"])
    start = text.index(GOOGL_START)
    end = text.index(GOOGL_END, start) + len(GOOGL_END)
    return text[start:end]


def main() -> None:
    """Write the JSONL file."""
    rows = [
        {
            "id": "ex21-apple-synthetic",
            "source": "synthetic Exhibit 21 (Apple-like)",
            "filing": APPLE,
            "focal": FOCAL_APPLE,
            "item": "EX-21",
            "text": EX21_TEXT,
            "entities": EX21_ENTITIES,
            "relationships": EX21_RELS,
        },
        {
            "id": "def14a-apple-synthetic",
            "source": "synthetic DEF 14A nominee table + bio sentence",
            "filing": APPLE,
            "focal": FOCAL_APPLE,
            "item": "DEF14A",
            "text": DEF14A_TEXT,
            "entities": DEF14A_ENTITIES,
            "relationships": DEF14A_RELS,
        },
        {
            "id": "item1-apple-synthetic",
            "source": "synthetic Item 1 prose",
            "filing": APPLE,
            "focal": FOCAL_APPLE,
            "item": "1",
            "text": ITEM1_TEXT,
            "entities": ITEM1_ENTITIES,
            "relationships": ITEM1_RELS,
        },
        {
            "id": "item1a-apple-synthetic",
            "source": "synthetic Item 1A prose",
            "filing": APPLE,
            "focal": FOCAL_APPLE,
            "item": "1A",
            "text": ITEM1A_TEXT,
            "entities": ITEM1A_ENTITIES,
            "relationships": ITEM1A_RELS,
        },
        {
            "id": "item1-googl-real",
            "source": "real 10-K excerpt, Alphabet 0001652044-26-000018 (Item 1, Google Services paragraph)",
            "filing": GOOGL,
            "focal": FOCAL_GOOGL,
            "item": "1",
            "text": googl_passage(),
            "entities": GOOGL_ENTITIES,
            "relationships": GOOGL_RELS,
        },
    ]
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} annotated chunks to {OUT}")


if __name__ == "__main__":
    main()
