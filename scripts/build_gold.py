"""Serialize the hand-written gold set to ``eval/gold.jsonl``.

Ground truth was written by hand from the fixture tables in
``src/graph/fixture.py`` (see ``docs/data_schema.md`` for scope). The script
only formats it; it never derives answers from the engine.

Run: python scripts/build_gold.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "eval" / "gold.jsonl"

# (id, question, expected_ids, hops, notes)
LOOKUP: list[tuple[str, str, list[str], int, str]] = [
    ("lk-01", "Who is the CEO of Apple?", ["tim_cook"], 1, "CEO_OF"),
    ("lk-02", "Who is the CEO of Microsoft?", ["satya_nadella"], 1, "CEO_OF"),
    ("lk-03", "Who leads NVIDIA?", ["jensen_huang"], 1, "CEO_OF"),
    ("lk-04", "Who is the CEO of Pfizer?", ["albert_bourla"], 1, "CEO_OF"),
    ("lk-05", "Which company does Lisa Su lead?", ["amd"], 1, "CEO_OF reverse"),
    ("lk-06", "Which company does Mary Barra lead?", ["gm"], 1, "CEO_OF reverse"),
    ("lk-07", "Who is the CEO of Eli Lilly?", ["david_ricks"], 1, "CEO_OF"),
    ("lk-08", "Who is the CEO of TSMC?", ["cc_wei"], 1, "CEO_OF, non-S&P filer"),
    ("lk-09", "Who is the parent company of LinkedIn?", ["microsoft"], 1, "SUBSIDIARY_OF"),
    ("lk-10", "Which company owns Instagram?", ["meta"], 1, "SUBSIDIARY_OF"),
    ("lk-11", "Which company owns Waymo?", ["google"], 1, "SUBSIDIARY_OF"),
    ("lk-12", "Who is the parent company of Optum?", ["unh"], 1, "SUBSIDIARY_OF"),
    (
        "lk-13",
        "Which products does NVIDIA offer?",
        ["prod_geforce", "prod_h100", "prod_cuda"],
        1,
        "OFFERS_PRODUCT",
    ),
    (
        "lk-14",
        "Which products does Eli Lilly offer?",
        ["prod_mounjaro", "prod_zepbound", "prod_verzenio"],
        1,
        "OFFERS_PRODUCT",
    ),
    ("lk-15", "Which company offers Keytruda?", ["merck"], 1, "OFFERS_PRODUCT reverse"),
    ("lk-16", "Which company makes Snapdragon?", ["qualcomm"], 1, "OFFERS_PRODUCT reverse"),
    (
        "lk-17",
        "Who are Apple's suppliers?",
        ["tsmc", "foxconn", "samsung", "broadcom", "qualcomm"],
        1,
        "SUPPLIED_BY out",
    ),
    (
        "lk-18",
        "Which companies are supplied by TSMC?",
        ["apple", "nvidia", "amd", "qualcomm", "broadcom", "intel"],
        1,
        "SUPPLIED_BY in",
    ),
    ("lk-19", "Which companies compete with Tesla?", ["ford", "gm"], 1, "COMPETES_WITH"),
    ("lk-20", "Which companies compete with Visa?", ["mastercard"], 1, "COMPETES_WITH"),
    (
        "lk-21",
        "Which risks does Tesla mention?",
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_climate_regulation",
            "risk_tariffs",
            "risk_talent",
        ],
        1,
        "MENTIONS_RISK",
    ),
    (
        "lk-22",
        "Which risks does JPMorgan mention?",
        ["risk_interest_rate", "risk_cybersecurity"],
        1,
        "MENTIONS_RISK",
    ),
    (
        "lk-23",
        "Which markets does Amazon operate in?",
        ["market_cloud", "market_digital_ads", "market_retail", "market_ai_models"],
        1,
        "OPERATES_IN",
    ),
    (
        "lk-24",
        "Which markets does Microsoft operate in?",
        ["market_cloud", "market_enterprise_software", "market_ai_models"],
        1,
        "OPERATES_IN",
    ),
    (
        "lk-25",
        "Which directors sit on Apple's board?",
        ["dir_castellanos"],
        1,
        "DIRECTOR_OF (synthetic director)",
    ),
    (
        "lk-26",
        "Which companies does Dana Whitfield serve as a director?",
        ["pfizer", "merck"],
        1,
        "DIRECTOR_OF reverse (synthetic)",
    ),
    (
        "lk-27",
        "Which subsidiaries does Meta have?",
        ["sub_instagram", "sub_whatsapp", "sub_facebook_tech"],
        1,
        "SUBSIDIARY_OF in",
    ),
    (
        "lk-28",
        "Which subsidiaries does Alphabet have?",
        ["sub_youtube", "sub_waymo", "sub_deepmind", "sub_google_ireland"],
        1,
        "SUBSIDIARY_OF in",
    ),
    (
        "lk-29",
        "What is Microsoft's ticker?",
        ["microsoft"],
        0,
        "property lookup; expected_text MSFT",
    ),
    ("lk-30", "Which sector is Pfizer in?", ["pfizer"], 0, "property lookup; expected_text Pharma"),
]
LOOKUP_TEXT = {"lk-29": "MSFT", "lk-30": "Pharma"}

TWO_HOP: list[tuple[str, str, list[str], str]] = [
    (
        "2h-01",
        "Who is the CEO of the parent company of LinkedIn?",
        ["satya_nadella"],
        "LinkedIn -SUBSIDIARY_OF-> Microsoft <-CEO_OF- Nadella",
    ),
    (
        "2h-02",
        "Who is the CEO of the company that owns Instagram?",
        ["mark_zuckerberg"],
        "Instagram -> Meta <- Zuckerberg",
    ),
    (
        "2h-03",
        "Who leads the parent company of Waymo?",
        ["sundar_pichai"],
        "Waymo -> Alphabet <- Pichai",
    ),
    (
        "2h-04",
        "Which risks are mentioned by the parent company of Waymo?",
        ["risk_cybersecurity", "risk_ai_regulation", "risk_antitrust", "risk_fx", "risk_talent"],
        "Alphabet's Item 1A risks",
    ),
    (
        "2h-05",
        "Which CEOs lead companies that compete with Ford?",
        ["elon_musk", "mary_barra"],
        "Ford competitors: Tesla, GM",
    ),
    (
        "2h-06",
        "Which CEOs lead competitors of JPMorgan?",
        ["brian_moynihan", "david_solomon"],
        "JPM competitors: BofA, Goldman",
    ),
    (
        "2h-07",
        "Which CEOs lead the competitors of Coca-Cola?",
        ["ramon_laguarta"],
        "KO competitor: PepsiCo",
    ),
    (
        "2h-08",
        "Which products are offered by competitors of AMD?",
        ["prod_geforce", "prod_h100", "prod_cuda", "prod_xeon", "prod_core"],
        "AMD competitors: NVIDIA, Intel",
    ),
    (
        "2h-09",
        "Which products are offered by competitors of Coca-Cola?",
        ["prod_pepsi", "prod_gatorade", "prod_doritos"],
        "PepsiCo products",
    ),
    (
        "2h-10",
        "Which risks are mentioned by companies supplied by TSMC?",
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_cybersecurity",
            "risk_ai_regulation",
            "risk_antitrust",
            "risk_fx",
            "risk_tariffs",
            "risk_talent",
        ],
        "union over Apple, NVIDIA, AMD, Qualcomm, Broadcom, Intel",
    ),
    (
        "2h-11",
        "Which suppliers do the competitors of NVIDIA use?",
        ["tsmc", "asml"],
        "AMD: TSMC; Intel: ASML, TSMC",
    ),
    (
        "2h-12",
        "Which suppliers do the competitors of Apple use?",
        ["asml", "nvidia", "broadcom"],
        "Samsung: ASML; Alphabet: NVIDIA, Broadcom",
    ),
    (
        "2h-13",
        "Which directors sit on boards of competitors of Pfizer?",
        ["dir_whitfield", "dir_oyelaran"],
        "Merck: Whitfield; J&J: Oyelaran (synthetic)",
    ),
    (
        "2h-14",
        "Which directors sit on the boards of competitors of Ford?",
        ["dir_okafor"],
        "GM: Okafor; Tesla has none (synthetic)",
    ),
    (
        "2h-15",
        "Which markets do the competitors of Walmart operate in?",
        ["market_retail", "market_cloud", "market_digital_ads", "market_ai_models"],
        "Costco + Amazon markets",
    ),
    (
        "2h-16",
        "Which subsidiaries belong to competitors of Coca-Cola?",
        ["sub_fritolay", "sub_quaker"],
        "PepsiCo Exhibit 21",
    ),
    (
        "2h-17",
        "Which subsidiaries belong to the competitors of Visa?",
        ["sub_mc_intl", "sub_vocalink"],
        "Mastercard Exhibit 21",
    ),
    (
        "2h-18",
        "Which risks are mentioned by companies whose CEO is Lisa Su?",
        ["risk_china_exposure", "risk_supply_concentration", "risk_talent"],
        "AMD risks",
    ),
    (
        "2h-19",
        "Which risks are mentioned by the company led by Jensen Huang?",
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_ai_regulation",
            "risk_tariffs",
            "risk_talent",
        ],
        "NVIDIA risks",
    ),
    (
        "2h-20",
        "Which products does the company led by Satya Nadella offer?",
        ["prod_azure", "prod_windows", "prod_m365", "prod_xbox"],
        "Microsoft products",
    ),
    (
        "2h-21",
        "Which companies share a supplier with NVIDIA?",
        ["apple", "amd", "qualcomm", "broadcom", "intel"],
        "via TSMC / Foxconn",
    ),
    (
        "2h-22",
        "Which companies share a supplier with Microsoft?",
        ["google", "meta", "amazon", "oracle"],
        "via NVIDIA / AMD",
    ),
    ("2h-23", "Which companies share a supplier with Ford?", ["apple", "gm"], "via Qualcomm"),
    ("2h-24", "Which companies share a market with Visa?", ["mastercard"], "Payments market"),
    (
        "2h-25",
        "Which companies share a market with Tesla?",
        ["ford", "gm"],
        "Electric vehicles market",
    ),
    (
        "2h-26",
        "Which directors sit on the boards of both Pfizer and Merck?",
        ["dir_whitfield"],
        "intersection of two 1-hop sets (synthetic)",
    ),
    ("2h-27", "Which companies are customers of both TSMC and ASML?", ["intel"], "intersection"),
    (
        "2h-28",
        "Which companies mention both China exposure and tariffs?",
        ["apple", "tesla", "ford", "gm", "walmart", "nvidia", "boeing", "micron"],
        "intersection of two risk sets",
    ),
    (
        "2h-29",
        "Which risks are shared by Ford and GM?",
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_climate_regulation",
            "risk_interest_rate",
            "risk_tariffs",
            "risk_commodity_prices",
        ],
        "intersection",
    ),
    (
        "2h-30",
        "Which companies compete with a company supplied by Foxconn?",
        ["samsung", "google", "amd", "intel"],
        "Foxconn customers Apple, NVIDIA -> competitors",
    ),
]

THREE_HOP: list[tuple[str, str, list[str], str]] = [
    (
        "3h-01",
        "Which CEOs lead companies that share a supplier with Apple?",
        [
            "jensen_huang",
            "lisa_su",
            "cristiano_amon",
            "hock_tan",
            "lip_bu_tan",
            "sundar_pichai",
            "mark_zuckerberg",
            "jim_farley",
            "mary_barra",
        ],
        "Apple -> suppliers -> other customers -> CEOs",
    ),
    (
        "3h-02",
        "Which CEOs lead companies that share a supplier with Microsoft?",
        ["sundar_pichai", "mark_zuckerberg", "andy_jassy", "safra_catz"],
        "via NVIDIA/AMD customers",
    ),
    (
        "3h-03",
        "Which products are offered by companies that share a supplier with NVIDIA?",
        [
            "prod_iphone",
            "prod_mac",
            "prod_ipad",
            "prod_apple_watch",
            "prod_ryzen",
            "prod_epyc",
            "prod_mi300",
            "prod_snapdragon",
            "prod_tomahawk",
            "prod_vcf",
            "prod_xeon",
            "prod_core",
        ],
        "Apple, AMD, Qualcomm, Broadcom, Intel products",
    ),
    (
        "3h-04",
        "Which directors sit on the boards of companies that compete with a company supplied by TSMC?",
        ["dir_petrov", "dir_abernathy", "dir_nakamura"],
        "TSMC customers -> competitors -> directors (synthetic)",
    ),
    (
        "3h-05",
        "Which risks are mentioned by competitors of the parent company of Instagram?",
        ["risk_cybersecurity", "risk_ai_regulation", "risk_antitrust", "risk_fx", "risk_talent"],
        "Instagram -> Meta -> Alphabet -> risks",
    ),
    (
        "3h-06",
        "Which subsidiaries belong to competitors of the parent company of Instagram?",
        ["sub_youtube", "sub_waymo", "sub_deepmind", "sub_google_ireland"],
        "Alphabet subsidiaries",
    ),
    (
        "3h-07",
        "Which CEOs lead competitors of the parent company of LinkedIn?",
        ["sundar_pichai", "andy_jassy", "safra_catz", "marc_benioff"],
        "Microsoft competitors' CEOs",
    ),
    (
        "3h-08",
        "Which markets do competitors of the parent company of Waymo operate in?",
        [
            "market_smartphones",
            "market_cloud",
            "market_enterprise_software",
            "market_ai_models",
            "market_digital_ads",
            "market_retail",
        ],
        "Apple, Microsoft, Amazon, Meta markets",
    ),
    (
        "3h-09",
        "Which risks are mentioned by companies that compete with the company led by Jensen Huang?",
        ["risk_china_exposure", "risk_supply_concentration", "risk_talent"],
        "AMD + Intel risks",
    ),
    (
        "3h-10",
        "Which products are offered by competitors of the company led by Jim Farley?",
        [
            "prod_model_3",
            "prod_model_y",
            "prod_cybertruck",
            "prod_megapack",
            "prod_silverado",
            "prod_escalade",
        ],
        "Tesla + GM products",
    ),
    (
        "3h-11",
        "Which directors sit on boards of companies that compete with the company led by Albert Bourla?",
        ["dir_whitfield", "dir_oyelaran"],
        "Pfizer competitors' directors (synthetic)",
    ),
    (
        "3h-12",
        "Which CEOs lead companies that share a risk with Eli Lilly?",
        ["albert_bourla", "joaquin_duato", "robert_davis", "robert_michael", "stephen_hemsley"],
        "drug pricing / patent expiry peers",
    ),
    (
        "3h-13",
        "Which subsidiaries belong to companies that share a market with Tesla?",
        ["sub_ford_credit", "sub_ford_canada", "sub_gm_financial", "sub_cruise"],
        "EV market peers' Exhibit 21",
    ),
    (
        "3h-14",
        "Which directors sit on the boards of companies that share a market with JPMorgan?",
        ["dir_kowalczyk"],
        "banking peers BofA, Goldman (synthetic)",
    ),
    (
        "3h-15",
        "Which products are offered by companies that share a supplier with Ford?",
        [
            "prod_iphone",
            "prod_mac",
            "prod_ipad",
            "prod_apple_watch",
            "prod_silverado",
            "prod_escalade",
        ],
        "Apple + GM via Qualcomm",
    ),
    (
        "3h-16",
        "Which risks are mentioned by companies that share a supplier with Ford?",
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_cybersecurity",
            "risk_ai_regulation",
            "risk_antitrust",
            "risk_fx",
            "risk_tariffs",
            "risk_climate_regulation",
            "risk_interest_rate",
            "risk_commodity_prices",
        ],
        "Apple + GM risks",
    ),
    (
        "3h-17",
        "Which CEOs lead companies that compete with a company supplied by Foxconn?",
        ["sundar_pichai", "lisa_su", "lip_bu_tan"],
        "Samsung has no CEO node",
    ),
    (
        "3h-18",
        "Which subsidiaries belong to companies that compete with a company supplied by Panasonic?",
        ["sub_ford_credit", "sub_ford_canada", "sub_gm_financial", "sub_cruise"],
        "Panasonic -> Tesla -> Ford, GM -> subsidiaries",
    ),
    (
        "3h-19",
        "Which markets do companies that share a supplier with Microsoft operate in?",
        [
            "market_cloud",
            "market_smartphones",
            "market_digital_ads",
            "market_ai_models",
            "market_retail",
            "market_enterprise_software",
        ],
        "Alphabet, Meta, Amazon, Oracle markets",
    ),
    (
        "3h-20",
        "Which directors sit on the boards of companies that share a supplier with Apple?",
        ["dir_nakamura", "dir_abernathy", "dir_petrov", "dir_mbeki", "dir_okafor"],
        "synthetic directors of NVIDIA, AMD/Intel, Alphabet, Meta, Ford/GM",
    ),
]

AGGREGATION: list[tuple[str, str, int, list[str], str]] = [
    (
        "ag-01",
        "How many companies mention China exposure as a risk?",
        15,
        [
            "apple",
            "microsoft",
            "nvidia",
            "amd",
            "intel",
            "qualcomm",
            "broadcom",
            "micron",
            "tesla",
            "boeing",
            "walmart",
            "tsmc",
            "asml",
            "ford",
            "gm",
        ],
        "MENTIONS_RISK in-degree",
    ),
    (
        "ag-02",
        "How many S&P 500 companies mention China exposure as a risk?",
        13,
        [
            "apple",
            "microsoft",
            "nvidia",
            "amd",
            "intel",
            "qualcomm",
            "broadcom",
            "micron",
            "tesla",
            "boeing",
            "walmart",
            "ford",
            "gm",
        ],
        "excludes TSMC, ASML",
    ),
    (
        "ag-03",
        "How many S&P 500 companies are supplied by TSMC?",
        6,
        ["apple", "nvidia", "amd", "qualcomm", "broadcom", "intel"],
        "",
    ),
    (
        "ag-04",
        "How many subsidiaries does Alphabet have?",
        4,
        ["sub_youtube", "sub_waymo", "sub_deepmind", "sub_google_ireland"],
        "",
    ),
    (
        "ag-05",
        "How many subsidiaries does Amazon have?",
        4,
        ["sub_aws", "sub_whole_foods", "sub_zappos", "sub_audible"],
        "",
    ),
    (
        "ag-06",
        "How many companies operate in the cloud infrastructure market?",
        4,
        ["microsoft", "google", "amazon", "oracle"],
        "",
    ),
    (
        "ag-07",
        "How many companies operate in the semiconductors market?",
        9,
        ["nvidia", "amd", "intel", "qualcomm", "broadcom", "micron", "tsmc", "samsung", "sk_hynix"],
        "",
    ),
    (
        "ag-08",
        "How many S&P 500 companies operate in the semiconductors market?",
        6,
        ["nvidia", "amd", "intel", "qualcomm", "broadcom", "micron"],
        "",
    ),
    (
        "ag-09",
        "How many companies compete with Microsoft?",
        4,
        ["google", "amazon", "oracle", "salesforce"],
        "undirected COMPETES_WITH",
    ),
    (
        "ag-10",
        "How many companies in the pharma sector mention drug pricing pressure?",
        5,
        ["pfizer", "jnj", "merck", "lilly", "abbvie"],
        "UnitedHealth is Healthcare, excluded",
    ),
    (
        "ag-11",
        "How many companies mention cybersecurity incidents?",
        16,
        [
            "apple",
            "microsoft",
            "google",
            "meta",
            "amazon",
            "oracle",
            "salesforce",
            "adobe",
            "jpm",
            "bofa",
            "goldman",
            "visa",
            "mastercard",
            "unh",
            "walmart",
            "costco",
        ],
        "",
    ),
    (
        "ag-12",
        "How many directors serve on more than one board?",
        14,
        [
            "dir_whitfield",
            "dir_oyelaran",
            "dir_castellanos",
            "dir_lindqvist",
            "dir_nakamura",
            "dir_okafor",
            "dir_rasmussen",
            "dir_petrov",
            "dir_sandoval",
            "dir_abernathy",
            "dir_kowalczyk",
            "dir_mbeki",
            "dir_haddad",
            "dir_ferreira",
        ],
        "degree >= 2 on DIRECTOR_OF (synthetic)",
    ),
    (
        "ag-13",
        "How many products does Tesla offer?",
        4,
        ["prod_model_3", "prod_model_y", "prod_cybertruck", "prod_megapack"],
        "",
    ),
    (
        "ag-14",
        "How many products does AbbVie offer?",
        4,
        ["prod_humira", "prod_skyrizi", "prod_rinvoq", "prod_botox"],
        "",
    ),
    (
        "ag-15",
        "How many risks does Apple mention?",
        7,
        [
            "risk_china_exposure",
            "risk_supply_concentration",
            "risk_cybersecurity",
            "risk_ai_regulation",
            "risk_antitrust",
            "risk_fx",
            "risk_tariffs",
        ],
        "",
    ),
    (
        "ag-16",
        "How many companies share a supplier with Apple?",
        9,
        ["nvidia", "amd", "qualcomm", "broadcom", "intel", "google", "meta", "ford", "gm"],
        "2-hop then count",
    ),
    (
        "ag-17",
        "How many companies are supplied by ASML?",
        5,
        ["intel", "micron", "tsmc", "samsung", "sk_hynix"],
        "",
    ),
    (
        "ag-18",
        "How many subsidiaries are incorporated in Ireland?",
        5,
        [
            "sub_apple_ops_intl",
            "sub_ms_ireland",
            "sub_google_ireland",
            "sub_adobe_ireland",
            "sub_allergan",
        ],
        "property filter, no seed",
    ),
    (
        "ag-19",
        "How many products are in the Vaccine category?",
        3,
        ["prod_comirnaty", "prod_prevnar", "prod_gardasil"],
        "property filter, no seed",
    ),
    (
        "ag-20",
        "How many CEOs lead companies that compete with Ford?",
        2,
        ["elon_musk", "mary_barra"],
        "2-hop then count",
    ),
]


def main() -> None:
    """Write ``eval/gold.jsonl`` (one JSON object per line)."""
    rows: list[dict[str, object]] = []
    for qid, q, ids, hops, notes in LOOKUP:
        row: dict[str, object] = {
            "id": qid,
            "category": "lookup",
            "question": q,
            "expected_ids": ids,
            "hops": hops,
            "notes": notes,
            "ground_truth": "manual (fixture tables)",
        }
        if qid in LOOKUP_TEXT:
            row["expected_text"] = LOOKUP_TEXT[qid]
        rows.append(row)
    for qid, q, ids, notes in TWO_HOP:
        rows.append(
            {
                "id": qid,
                "category": "two_hop",
                "question": q,
                "expected_ids": ids,
                "hops": 2,
                "notes": notes,
                "ground_truth": "manual (fixture tables)",
            }
        )
    for qid, q, ids, notes in THREE_HOP:
        rows.append(
            {
                "id": qid,
                "category": "three_hop",
                "question": q,
                "expected_ids": ids,
                "hops": 3,
                "notes": notes,
                "ground_truth": "manual (fixture tables)",
            }
        )
    for qid, q, count, ids, notes in AGGREGATION:
        rows.append(
            {
                "id": qid,
                "category": "aggregation",
                "question": q,
                "expected_count": count,
                "expected_ids": ids,
                "hops": 2,
                "notes": notes,
                "ground_truth": "manual (fixture tables)",
            }
        )
    assert len(rows) == 100, len(rows)
    for r in AGGREGATION:
        assert r[2] == len(r[3]), r[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
