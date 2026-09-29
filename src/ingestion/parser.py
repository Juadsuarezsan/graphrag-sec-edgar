"""HTML parsing for SEC filings (10-K bodies, Exhibit 21 tables, DEF 14A).

The previous regex-based stripper lost the ``Item 1. Business`` header on
filers that wrap it in nested ``<span><font>`` runs or use ``&nbsp;``. This
parser walks the DOM with BeautifulSoup + lxml, drops hidden inline-XBRL
headers, keeps inline formatting on one line and breaks only on block tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "tr",
        "li",
        "ul",
        "ol",
        "table",
        "thead",
        "tbody",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "title",
        "body",
        "html",
    }
)
_CELL_TAGS = frozenset({"td", "th"})
_SKIP_TAGS = frozenset({"script", "style", "head", "ix:header", "noscript"})
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\r\f\v​]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def _walk(node: Tag | NavigableString, out: list[str]) -> None:
    if isinstance(node, Comment):
        return
    if isinstance(node, NavigableString):
        out.append(str(node))
        return
    name = (node.name or "").lower()
    if name in _SKIP_TAGS:
        return
    style = node.get("style")
    if isinstance(style, str) and _HIDDEN_RE.search(style):
        return
    is_block = name in _BLOCK_TAGS
    if is_block:
        out.append("\n")
    for child in node.children:
        if isinstance(child, Tag | NavigableString):
            _walk(child, out)
    if name in _CELL_TAGS:
        out.append(" | ")
    if is_block:
        out.append("\n")


def html_to_text(html: str) -> str:
    """Convert filing HTML to normalised plain text.

    Guarantees:
      * hidden XBRL headers and ``display:none`` blocks are dropped;
      * inline runs (``<span><font>``) stay on the same line;
      * ``&nbsp;`` and zero-width characters become plain spaces;
      * table cells are separated by `` | `` and rows by newlines.
    """
    if not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    root = soup.body or soup
    _walk(root, out)
    text = "".join(out).replace("\xa0", " ").replace(" ", " ")
    lines = [_WS_RE.sub(" ", line).strip(" |") for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return _MULTI_NL_RE.sub("\n", text).strip()


# ---------------------------------------------------------------------------
# Exhibit 21 — subsidiaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsidiaryRow:
    """One subsidiary line from Exhibit 21."""

    name: str
    jurisdiction: str | None
    ownership_pct: float | None = None


_HEADER_WORDS = {
    "name",
    "subsidiary",
    "subsidiaries",
    "jurisdiction",
    "entity",
    "state",
    "country",
    "organization",
    "incorporation",
    "ownership",
    "%",
}
_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_PAREN_JURIS_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<juris>[A-Z][A-Za-z .'-]{1,40})\)\s*$")
_DASH_JURIS_RE = re.compile(r"^(?P<name>.+?)\s+[—–-]\s+(?P<juris>[A-Z][A-Za-z .'-]{1,40})$")


def _is_header(cells: list[str]) -> bool:
    words = {w for c in cells for w in c.lower().replace("/", " ").split()}
    return bool(words) and words <= _HEADER_WORDS


def _clean_name(name: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", name.strip().strip("*").strip())


def parse_exhibit21(html_or_text: str) -> list[SubsidiaryRow]:
    """Extract subsidiaries from an Exhibit 21 document.

    Accepts HTML (tables) or the plain text produced by :func:`html_to_text`.
    Rows whose first cell is a header word, footnote or blank are skipped.
    """
    text = html_to_text(html_or_text) if "<" in html_or_text else html_or_text
    rows: list[SubsidiaryRow] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        cells = [c.strip() for c in line.split(" | ") if c.strip()]
        if not cells or _is_header(cells):
            continue
        if len(cells) >= 2:
            name, juris = _clean_name(cells[0]), cells[1].strip()
            pct_m = _PCT_RE.search(" ".join(cells[2:])) or (_PCT_RE.search(juris))
            pct = float(pct_m.group(1)) if pct_m else None
            if _PCT_RE.fullmatch(juris):
                juris = ""
        else:
            m = _PAREN_JURIS_RE.match(cells[0]) or _DASH_JURIS_RE.match(cells[0])
            if not m:
                continue
            name, juris, pct = _clean_name(m.group("name")), m.group("juris").strip(), None
        if len(name) < 3 or name.lower() in _HEADER_WORDS or name.lower().startswith("exhibit"):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(SubsidiaryRow(name=name, jurisdiction=juris or None, ownership_pct=pct))
    return rows


# ---------------------------------------------------------------------------
# DEF 14A — directors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectorRow:
    """One director from a proxy statement."""

    name: str
    age: int | None
    director_since: int | None
    role: str | None = None


_DIRECTOR_SENTENCE_RE = re.compile(
    r"(?P<name>[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3}),\s*(?:age\s+)?(?P<age>\d{2}),?\s+"
    r"(?:has\s+(?:served|been)\s+(?:as\s+)?(?:a\s+)?(?:member\s+of\s+(?:our|the)\s+board|director)"
    r"(?:\s+of\s+(?:the\s+)?(?:Company|Board))?\s+since\s+(?P<since>(?:19|20)\d{2}))",
    re.IGNORECASE,
)
_ROLE_WORDS = ("Chair", "Lead Independent Director", "Chief Executive Officer", "CEO", "President")


def _row_role(cells: list[str]) -> str | None:
    joined = " ".join(cells)
    for role in _ROLE_WORDS:
        if role.lower() in joined.lower():
            return role
    return None


def parse_def14a_directors(html_or_text: str) -> list[DirectorRow]:
    """Extract director nominees from a DEF 14A.

    Two strategies are combined: nominee tables whose header has ``Name`` and
    ``Age`` (or ``Director Since``), and biography sentences such as
    ``Jane Doe, 61, has served as a director since 2015``.
    """
    text = html_to_text(html_or_text) if "<" in html_or_text else html_or_text
    found: dict[str, DirectorRow] = {}
    in_table = False
    for line in text.split("\n"):
        cells = [c.strip() for c in line.split(" | ") if c.strip()]
        lowered = [c.lower() for c in cells]
        if (
            len(cells) >= 2
            and "name" in lowered
            and any(h in lowered for h in ("age", "director since", "since"))
        ):
            in_table = True
            continue
        if in_table:
            if len(cells) < 2:
                in_table = False
                continue
            name = _clean_name(cells[0])
            nums = [int(c) for c in cells[1:] if c.isdigit()]
            age = next((n for n in nums if 25 <= n <= 99), None)
            since = next((n for n in nums if 1900 <= n <= 2100), None)
            if re.match(r"^[A-Z][a-zA-Z.'-]+(\s+[A-Z][a-zA-Z.'-]+){1,3}$", name):
                found.setdefault(
                    name.lower(),
                    DirectorRow(name=name, age=age, director_since=since, role=_row_role(cells)),
                )
    for m in _DIRECTOR_SENTENCE_RE.finditer(text):
        name = m.group("name").strip()
        found.setdefault(
            name.lower(),
            DirectorRow(name=name, age=int(m.group("age")), director_since=int(m.group("since"))),
        )
    return list(found.values())
