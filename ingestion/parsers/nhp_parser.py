"""
Parser for National Health Portal (NHP) .txt files.

These files were scraped from nhp.gov.in and contain:
  - A 4-line metadata header (SOURCE, TITLE, SOURCE_TAG, SCRAPED)
  - A large MENU navigation block (site-wide navigation, not content)
  - The actual disease content with sections
  - A footer with "Related Pages", validation info, and discussion boilerplate

This parser strips all noise and returns clean disease-information text.
"""

import re
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def parse_nhp_txt(file_path: Path) -> Dict[str, str]:
    """
    Parse a single NHP .txt file and return clean content.

    Args:
        file_path: Path to the .txt file.

    Returns:
        dict with keys:
            - "title": Document title extracted from the header.
            - "full_text": Cleaned disease content (no header, menu, footer).
    """
    raw_text = file_path.read_text(encoding="utf-8")

    # ── Step 1: Extract title from the metadata header ────────────────
    title = _extract_title(raw_text)

    # ── Step 2: Strip the metadata header (first 4 lines) ─────────────
    text = _strip_metadata_header(raw_text)

    # ── Step 3: Strip the MENU navigation block ───────────────────────
    text = _strip_menu_block(text)

    # ── Step 4: Strip the footer boilerplate ──────────────────────────
    text = _strip_footer(text)

    # ── Step 5: Clean up whitespace ───────────────────────────────────
    text = _normalize_whitespace(text)

    logger.info(f"Parsed NHP file: {file_path.name} -> title='{title}', "
                f"text_length={len(text)} chars")

    return {"title": title, "full_text": text}


def _extract_title(raw_text: str) -> str:
    """Extract TITLE from the 4-line metadata header."""
    match = re.search(r"^TITLE:\s*(.+)$", raw_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fallback: use filename-like guess from SOURCE line
    match = re.search(r"SOURCE:.*/([^/]+)$", raw_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Unknown NHP Document"


def _strip_metadata_header(raw_text: str) -> str:
    """
    Remove the 4-line metadata header:
      SOURCE: ...
      TITLE: ...
      SOURCE_TAG: ...
      SCRAPED: ...
    plus the blank line that follows.
    """
    lines = raw_text.split("\n")
    # Find where the header ends: after SCRAPED line + blank line
    cut_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("SCRAPED:"):
            cut_idx = i + 2  # skip SCRAPED line and the following blank line
            break
    return "\n".join(lines[cut_idx:])


def _strip_menu_block(text: str) -> str:
    """
    Remove the MENU navigation block.
    Starts with "MENU" and ends with "Close Menu".
    Everything between is site navigation, not disease content.
    """
    # Pattern: "MENU" through "Close Menu" (inclusive)
    menu_pattern = r"^MENU\s*$.*?^Close Menu\s*$"
    text = re.sub(menu_pattern, "", text, flags=re.MULTILINE | re.DOTALL)
    return text


def _strip_footer(text: str) -> str:
    """
    Remove footer boilerplate that appears after the actual content:
      - "Related Pages" and everything after it
      - "CREATED / VALIDATED BY" block
      - "Discussion" block (login/signup/spam prevention)
      - "The content on this page has been supervised..." line
    """
    # Cut at "Related Pages" — everything after is navigation links, not content
    text = re.split(r"^Related Pages\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]

    # Remove "CREATED / VALIDATED BY" through "LAST UPDATED ON" lines
    text = re.sub(
        r"^CREATED / VALIDATED BY\s*$.*?^LAST UPDATED ON\s*$",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Remove "Discussion" block (starts with "Discussion" and runs to end)
    text = re.split(r"^Discussion\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]

    # Remove trailing validation/supervision line
    text = re.sub(
        r"^The content on this page has been supervised.*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines and strip leading/trailing whitespace."""
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text