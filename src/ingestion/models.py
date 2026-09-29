"""Typed records exchanged along the ingestion pipeline."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

FormType = Literal["10-K", "10-Q", "DEF 14A", "EX-21"]

#: Section identifiers the chunker recognises. ``EX-21`` and ``DEF14A`` are
#: pseudo-sections for structured exhibits/proxies.
SectionItem = Literal["1", "1A", "1B", "1C", "2", "3", "7", "7A", "8", "EX-21", "DEF14A", "FULL"]

SECTION_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "EX-21": "Subsidiaries of the Registrant",
    "DEF14A": "Proxy Statement - Directors",
    "FULL": "Whole document",
}


class Filing(BaseModel):
    """One SEC filing (metadata only; text lives in sections/chunks)."""

    ticker: str = Field(min_length=1, max_length=12)
    cik: str = Field(min_length=1, max_length=10)
    company_name: str = Field(min_length=1)
    accession: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    form_type: FormType = "10-K"
    filing_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_url: str | None = None

    @property
    def company_id(self) -> str:
        """Stable graph id for the filer."""
        return self.ticker.lower().replace(".", "_")


class Section(BaseModel):
    """A contiguous 10-K section."""

    item: SectionItem
    title: str
    text: str
    char_start: int = 0
    char_end: int = 0

    @property
    def n_chars(self) -> int:
        """Length of the section text."""
        return len(self.text)


class Chunk(BaseModel):
    """A piece of a section small enough for one extraction call."""

    id: str
    filing: Filing
    item: SectionItem
    index: int
    text: str
    char_start: int
    char_end: int

    @classmethod
    def make(
        cls, filing: Filing, item: SectionItem, index: int, text: str, start: int, end: int
    ) -> Chunk:
        """Build a chunk with a deterministic content-addressed id."""
        digest = hashlib.sha1(
            f"{filing.accession}|{item}|{index}|{text[:64]}".encode()
        ).hexdigest()[:12]
        return cls(
            id=f"{filing.ticker.lower()}-{item.lower()}-{index:03d}-{digest}",
            filing=filing,
            item=item,
            index=index,
            text=text,
            char_start=start,
            char_end=end,
        )
