"""Plain-text guideline parser: normalise, detect headings, split into sections."""

from __future__ import annotations

import re

from app.documents.base import ParsedDocument, Section

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")
_MD_HEADING = re.compile(r"^(#{1,6})\s+\S")
# Sub-headings inside a numbered section ("Approach to solve:", "Sample reply:") are not
# section boundaries; only top-level headings split sections.
_SUBHEADING_MAX_WORDS = 8


def normalize_text(text: str) -> str:
    """Normalise newlines and strip trailing whitespace; keep content otherwise intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def heading_level(line: str) -> int | None:
    """Return a heading level (1 = top) or None. Numbered `1.` / ALL CAPS / `#` are headings;
    short lines ending with ':' are sub-headings (level 9) that never start a section."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if m := _MD_HEADING.match(s):
        return len(m.group(1))
    if (m := _NUMBERED.match(s)) and len(s) <= 70 and not s.endswith((".", "?", "!")):
        return m.group(1).count(".") + 1
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and s == s.upper() and not s.endswith((".", ",")):
        return 1
    if s.endswith(":") and len(s) <= 60 and len(s.split()) <= _SUBHEADING_MAX_WORDS:
        return 9
    return None


def is_heading(line: str) -> bool:
    return heading_level(line) is not None


def clean_heading(line: str) -> str:
    s = line.strip().lstrip("#").strip().rstrip(":").strip()
    return re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", s)


class TxtParser:
    format = "txt"

    def parse(self, text: str) -> ParsedDocument:
        text = normalize_text(text)
        lines = text.split("\n")
        # (offset, level, heading) for section-starting headings (level < 9)
        starts: list[tuple[int, int, str]] = []
        offset = 0
        top_level: int | None = None
        for line in lines:
            level = heading_level(line)
            if level is not None and level < 9:
                top_level = level if top_level is None else min(top_level, level)
                starts.append((offset, level, clean_heading(line)))
            offset += len(line) + 1

        sections: list[Section] = []
        if not starts or starts[0][0] > 0:
            # preamble before the first heading (or a document without headings)
            end = starts[0][0] if starts else len(text)
            pre = text[:end].strip()
            if pre:
                sections.append(_section(len(sections), None, None, pre, 0, end))
        stack: list[tuple[int, str]] = []
        for i, (start, level, heading) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            path = " > ".join(h for _, h in stack)
            body = text[start:end].strip()
            sections.append(_section(len(sections), heading, path, body, start, end))
        return ParsedDocument(text=text, sections=sections)


def _section(
    ordinal: int, heading: str | None, path: str | None, text: str, start: int, end: int
) -> Section:
    return Section(
        ref=f"S{ordinal + 1}",
        ordinal=ordinal,
        heading=heading,
        path=path,
        text=text,
        char_start=start,
        char_end=end,
    )
