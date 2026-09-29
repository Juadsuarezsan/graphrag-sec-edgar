"""Section-aware chunking of 10-K text.

A 10-K lists every item twice: once in the table of contents and once as the
real heading. For each item we therefore keep the occurrence whose body (text
up to the next heading) is longest, which is always the real one.
"""

from __future__ import annotations

import re
from typing import Final

from loguru import logger

from src.ingestion.models import SECTION_TITLES, Chunk, Filing, Section, SectionItem

_ITEM_RE: Final = re.compile(
    r"^\s*(?:PART\s+[IV]+\s*[.,:\-—–]?\s*)?ITEM\s*(?P<item>\d{1,2}[ABC]?)\s*[.:\-—–]?\s*(?P<title>[^\n]{0,90})$",
    re.IGNORECASE | re.MULTILINE,
)
_WANTED: Final[tuple[SectionItem, ...]] = ("1", "1A", "1B", "1C", "2", "3", "7", "7A", "8")
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _all_headings(text: str) -> list[tuple[str, int, int]]:
    """Return ``(item, header_start, body_start)`` for every ``Item N`` line."""
    out: list[tuple[str, int, int]] = []
    for m in _ITEM_RE.finditer(text):
        item = m.group("item").upper()
        out.append((item, m.start(), m.end()))
    return out


def split_sections(
    text: str, wanted: tuple[SectionItem, ...] = _WANTED, min_chars: int = 200
) -> list[Section]:
    """Split plain 10-K text into the requested items.

    Args:
        text: Output of :func:`~src.ingestion.parser.html_to_text`.
        wanted: Items to return, in document order.
        min_chars: Bodies shorter than this are treated as TOC entries.

    Returns:
        Sections in document order. When no heading is found the whole text is
        returned as a single ``FULL`` section so downstream never gets nothing.
    """
    headings = _all_headings(text)
    if not headings:
        logger.warning("no Item headings found; returning FULL section ({} chars)", len(text))
        return [
            Section(
                item="FULL",
                title=SECTION_TITLES["FULL"],
                text=text.strip(),
                char_start=0,
                char_end=len(text),
            )
        ]

    best: dict[str, tuple[int, int, int]] = {}  # item -> (body_start, body_end, length)
    for i, (item, _hstart, body_start) in enumerate(headings):
        body_end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        length = body_end - body_start
        if item in wanted and length >= min_chars and length > best.get(item, (0, 0, -1))[2]:
            best[item] = (body_start, body_end, length)

    sections: list[Section] = []
    for item, (start, end, _len) in sorted(best.items(), key=lambda kv: kv[1][0]):
        body = text[start:end].strip()
        # The heading line may carry the title ("Item 1. Business"): drop a leading title echo.
        title = SECTION_TITLES.get(item, item)
        if body.lower().startswith(title.lower()):
            body = body[len(title) :].lstrip(" .:-—–\n")
        sections.append(Section(item=item, title=title, text=body, char_start=start, char_end=end))
    if not sections:
        logger.warning("headings found but every body < {} chars; returning FULL", min_chars)
        return [
            Section(
                item="FULL",
                title=SECTION_TITLES["FULL"],
                text=text.strip(),
                char_start=0,
                char_end=len(text),
            )
        ]
    return sections


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[tuple[str, int, int]]:
    """Split text into ``(chunk, start, end)`` windows on sentence boundaries.

    Args:
        text: Section text.
        max_chars: Hard cap per chunk.
        overlap: Characters of trailing context repeated at the start of the
            next chunk (clamped below ``max_chars``).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = min(overlap, max_chars // 2)
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    # sentence boundaries as absolute offsets
    bounds = [m.end() for m in _SENTENCE_END.finditer(text)] + [len(text)]
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        limit = start + max_chars
        if limit >= len(text):
            end = len(text)
        else:
            candidates = [b for b in bounds if start < b <= limit]
            end = candidates[-1] if candidates else limit
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_sections(
    filing: Filing, sections: list[Section], max_chars: int = 2000, overlap: int = 200
) -> list[Chunk]:
    """Chunk every section of a filing into :class:`Chunk` records."""
    out: list[Chunk] = []
    for sec in sections:
        for i, (piece, s, e) in enumerate(
            chunk_text(sec.text, max_chars=max_chars, overlap=overlap)
        ):
            out.append(
                Chunk.make(filing, sec.item, i, piece, sec.char_start + s, sec.char_start + e)
            )
    logger.info(
        "{} {}: {} sections -> {} chunks", filing.ticker, filing.form_type, len(sections), len(out)
    )
    return out
