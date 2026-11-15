"""Document parsing interface. Every format is parsed into the same `ParsedDocument`, so
compilation never sees the raw format. Only `.txt` is implemented; PDF/DOCX plug in here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Section:
    ref: str  # "S1", "S2", ... stable within one version
    ordinal: int
    heading: str | None
    path: str | None  # e.g. "Billing > Refunds" when headings nest
    text: str  # full section text including its heading line
    char_start: int
    char_end: int

    @property
    def body(self) -> str:
        """Section text without its heading line."""
        if self.heading is None:
            return self.text
        return self.text.split("\n", 1)[1].strip() if "\n" in self.text else ""


@dataclass
class ParsedDocument:
    text: str  # normalised full text
    sections: list[Section] = field(default_factory=list)

    def by_ref(self) -> dict[str, Section]:
        return {s.ref: s for s in self.sections}


class DocumentParser(Protocol):
    format: str  # "txt"

    def parse(self, text: str) -> ParsedDocument: ...
