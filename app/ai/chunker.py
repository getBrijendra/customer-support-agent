"""Guideline text normalisation, heading detection and chunking."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
_MD_HEADING = re.compile(r"^#{1,6}\s+\S")

# Split on headings first, then blank lines, lines, sentences, words.
SEPARATORS = [
    r"\n(?=#{1,6} )",  # before a markdown heading
    r"\n(?=\d+(?:\.\d+)*[.)]?\s+[A-Z])",  # before a numbered heading
    r"\n(?=[A-Z][A-Z0-9 &/,'()\-]{3,}:?\n)",  # before an ALL-CAPS heading line
    r"\n\n",
    r"\n",
    r"(?<=[.!?])\s",
    r" ",
    "",
]


def normalize_text(text: str) -> str:
    """Normalise newlines and strip trailing whitespace; keep content otherwise intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _MD_HEADING.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and s == s.upper() and not s.endswith((".", ",")):
        return True
    if _NUMBERED.match(s) and len(s) <= 60 and not s.endswith((".", "?", "!")):
        return True
    return s.endswith(":") and len(s) <= 60 and len(s.split()) <= 8


def clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip().rstrip(":").strip()


def find_headings(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, heading_text) for every heading line, in order."""
    out: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        if is_heading(line):
            out.append((offset, clean_heading(line)))
        offset += len(line) + 1
    return out


def make_chunk_id(version_id: Any, index: int) -> str:
    return f"{version_id}:{index:04d}"


@dataclass
class Chunk:
    id: str
    document: Document


def chunk_guideline(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    version_id: Any,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split normalised guideline text into chunks with deterministic ids and section metadata."""
    text = normalize_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator="start",
        add_start_index=True,
    )
    headings = find_headings(text)
    offsets = [h[0] for h in headings]
    docs = splitter.create_documents([text])
    chunks: list[Chunk] = []
    for i, doc in enumerate(docs):
        start = int(doc.metadata.get("start_index", 0))
        # Nearest heading at or before the chunk start (or the chunk's own first line).
        pos = bisect.bisect_right(offsets, max(start, 0)) - 1
        first_line = doc.page_content.split("\n", 1)[0]
        section = (
            clean_heading(first_line)
            if is_heading(first_line)
            else (headings[pos][1] if pos >= 0 else None)
        )
        chunk_id = make_chunk_id(version_id, i)
        metadata: dict[str, Any] = {
            **(base_metadata or {}),
            "chunk_index": i,
            "chunk_id": chunk_id,
            "section": section or "",
            "start_index": start,
        }
        chunks.append(
            Chunk(
                id=chunk_id,
                document=Document(id=chunk_id, page_content=doc.page_content, metadata=metadata),
            )
        )
    return chunks
