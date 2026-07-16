"""
Parser for PubMed research paper PDFs using Unstructured.io.

These are native-text (born-digital) PDFs with standard IMRaD structure:
  - Title / Author block
  - Abstract (often with Background/Objective/Methods/Results sub-sections)
  - Introduction
  - Methods / Materials and Methods
  - Results
  - Discussion
  - Conclusions (optional)
  - Acknowledgments (boilerplate — stripped)
  - Author Contributions (boilerplate — stripped)
  - References (stripped — not useful for clinical RAG)

This parser:
  1. Extracts text using Unstructured's partition_pdf
  2. Detects sections via regex
  3. Strips References, Acknowledgments, Author Contributions
  4. Returns a list of sections with headings and content
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

# ── Section detection patterns ──────────────────────────────────────────
# Ordered from most specific to least, to avoid false matches.
# Each tuple: (regex_pattern, canonical_heading_name)
SECTION_PATTERNS: List[tuple] = [
    # Abstract variants
    (r"^(?:Abstract|ABSTRACT)\s*$", "Abstract"),
    (r"^(?:Background|BACKGROUND)\s*$", "Abstract"),  # standalone Background → fold into Abstract
    # Introduction
    (r"^(?:Introduction|INTRODUCTION)\s*$", "Introduction"),
    # Methods variants
    (r"^(?:Methods|METHODS|Materials?\s*(?:and|&)\s*Methods?|MATERIALS?\s*(?:AND|&)\s*METHODS?|Methodology|METHODOLOGY)\s*$", "Methods"),
    # Results
    (r"^(?:Results|RESULTS|Findings|FINDINGS)\s*$", "Results"),
    # Discussion
    (r"^(?:Discussion|DISCUSSION)\s*$", "Discussion"),
    # Conclusions (optional, fold into Discussion if standalone)
    (r"^(?:Conclusions?|CONCLUSIONS?|Summary|SUMMARY)\s*$", "Conclusions"),
    # Sections to STRIP (not clinical content)
    (r"^(?:Acknowledggments?|ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?)\s*$", "__STRIP__"),
    (r"^(?:Author\s*Contributions?|AUTHOR\s*CONTRIBUTIONS?)\s*$", "__STRIP__"),
    (r"^(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature\s*Cited|LITERATURE\s*CITED)\s*$", "__STRIP__"),
    (r"^(?:Funding|FUNDING|Grant\s*Support|GRANT\s*SUPPORT)\s*$", "__STRIP__"),
    (r"^(?:Conflict\s*of\s*Interest|CONFLICT\s*OF\s*INTEREST|Competing\s*Interests?|COMPETING\s*INTERESTS?|Declaration|DECLARATION)\s*$", "__STRIP__"),
    (r"^(?:Supplementary\s*(?:Materials?|Information|Data)|SUPPLEMENTARY\s*(?:MATERIALS?|INFORMATION|DATA)|Supporting\s*Information|SUPPORTING\s*INFORMATION)\s*$", "__STRIP__"),
    (r"^(?:Data\s*Availability|DATA\s*AVAILABILITY|Data\s*Access|DATA\s*ACCESS)\s*$", "__STRIP__"),
]


def parse_pubmed_pdf(file_path: Path) -> Dict:
    """
    Parse a PubMed PDF and return structured sections.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        dict with keys:
            - "title": Document title extracted from first page.
            - "sections": List of {"heading": str, "content": str} dicts,
              in document order, with References/Acknowledgments/etc. stripped.
    """
    logger.info(f"Parsing PubMed PDF: {file_path.name}")

    # ── Step 1: Extract text using Unstructured ────────────────────────
    elements = partition_pdf(
        filename=str(file_path),
        strategy="fast",  # "fast" for born-digital PDFs (not OCR)
        include_page_breaks=False,
    )

    # Combine all elements into a single text block, preserving paragraph breaks
    full_text = "\n\n".join(
        str(el).strip() for el in elements if str(el).strip()
    )

    # ── Step 2: Extract title ──────────────────────────────────────────
    title = _extract_title(full_text, file_path)

    # ── Step 3: Detect sections ────────────────────────────────────────
    sections = _segment_into_sections(full_text)

    # ── Step 4: Strip unwanted sections ────────────────────────────────
    sections = _strip_unwanted_sections(sections)

    # ── Step 5: Clean section content ──────────────────────────────────
    for sec in sections:
        sec["content"] = _clean_section_text(sec["content"])

    logger.info(f"Parsed {file_path.name}: title='{title}', "
                f"{len(sections)} sections retained")

    return {"title": title, "sections": sections}


def _extract_title(full_text: str, file_path: Path) -> str:
    """
    Extract the paper title from the beginning of the text.
    The title is typically the first substantial line(s) before the author block.
    """
    lines = full_text.strip().split("\n")
    # Skip leading noise like "RESEARCH ARTICLE", journal headers, DOI lines
    skip_patterns = [
        r"^(?:RESEARCH\s*ARTICLE|REVIEW\s*ARTICLE|ORIGINAL\s*ARTICLE|CASE\s*REPORT)",
        r"^PLOS\s*ONE",
        r"^https?://doi",
        r"^a1111111111$",
        r"^OPEN\s*ACCESS",
        r"^Citation:",
        r"^Editor:",
        r"^Received:",
        r"^Accepted:",
        r"^Published:",
        r"^Copyright:",
        r"^Hindawi$",
        r"^BioMed\s*Research\s*International$",
        r"^Volume\s*\d+",
    ]
    skip_re = re.compile("|".join(skip_patterns), re.IGNORECASE)

    title_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            if title_lines:
                break  # blank line after title block → stop
            continue
        if skip_re.match(stripped):
            continue
        # If we've already collected title lines and hit something that looks
        # like an author affiliation (numbers, commas, @), stop.
        if title_lines and re.search(r"\b\d{1,2}\b.*@|department|university|college|institute", stripped, re.IGNORECASE):
            break
        title_lines.append(stripped)
        # Most titles are 1-3 lines; stop if we have enough
        if len(title_lines) >= 3:
            break

    if title_lines:
        return " ".join(title_lines).strip()
    # Fallback: use filename
    stem = file_path.stem
    # Clean up PMC prefix and any trailing whitespace in filename
    return stem.strip().replace("_", " ")


def _segment_into_sections(full_text: str) -> List[Dict[str, str]]:
    """
    Split full_text into sections based on detected headings.
    Returns list of {"heading": str, "content": str} in document order.
    """
    lines = full_text.split("\n")
    section_boundaries = []  # list of (line_index, canonical_heading)

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern, canonical in SECTION_PATTERNS:
            if re.match(pattern, stripped):
                section_boundaries.append((i, canonical))
                break  # first match wins

    # Build sections
    sections = []
    # Everything before the first detected heading is "Preamble" (title/authors/etc.)
    if section_boundaries:
        first_idx, _ = section_boundaries[0]
        preamble = "\n".join(lines[:first_idx]).strip()
        if preamble:
            sections.append({"heading": "Preamble", "content": preamble})

    for j, (start_idx, heading) in enumerate(section_boundaries):
        # Content starts AFTER the heading line
        content_start = start_idx + 1
        # Content ends at the next section boundary (or EOF)
        if j + 1 < len(section_boundaries):
            content_end = section_boundaries[j + 1][0]
        else:
            content_end = len(lines)
        content = "\n".join(lines[content_start:content_end]).strip()
        sections.append({"heading": heading, "content": content})

    # If no sections detected at all, return entire text as one section
    if not sections:
        sections.append({"heading": "Full Text", "content": full_text.strip()})

    return sections


def _strip_unwanted_sections(sections: List[Dict]) -> List[Dict]:
    """
    Remove sections marked __STRIP__ (References, Acknowledgments, etc.).
    Also, once we hit a __STRIP__ section, everything after it is also stripped
    (since References is always the last meaningful section, and everything
    after it — like Funding, Conflicts — is boilerplate).
    """
    cleaned = []
    for sec in sections:
        if sec["heading"] == "__STRIP__":
            # Stop collecting — everything from here on is stripped
            break
        cleaned.append(sec)
    return cleaned


def _clean_section_text(text: str) -> str:
    """Clean up section text: remove citation markers, normalize whitespace."""
    # Remove inline citation numbers like [1], [1,2], [1-3], [1, 2, 3]
    text = re.sub(r"\[\d+(?:[,\-]\d+)*\]", "", text)
    # Remove standalone reference numbers at line starts
    text = re.sub(r"^\d{1,3}\.\s*", "", text, flags=re.MULTILINE)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()