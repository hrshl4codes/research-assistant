"""
Document loader. Extracts text and page metadata from PDFs and plain text files.

Uses PyMuPDF (fitz) for PDFs. Returns one RawPage per non-blank page, with page
numbers intact. Blank pages are skipped so they don't produce empty chunks that
hurt search quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class RawPage:
    """A single page extracted from a source document."""

    text: str
    source: str       # filename, e.g. "paper.pdf"
    page_number: int  # 1-indexed for human-readable citations


def load_pdf(path: Union[Path, str]) -> list[RawPage]:
    """
    Extract text from every non-blank page of a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        One RawPage per non-blank page, in document order.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file cannot be opened as a PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path}': {exc}") from exc

    pages: list[RawPage] = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            logger.debug("Skipping blank page %d in '%s'.", i + 1, path.name)
            continue
        pages.append(RawPage(text=text, source=path.name, page_number=i + 1))

    doc.close()
    logger.info("Loaded %d non-blank page(s) from '%s'.", len(pages), path.name)
    return pages


def load_text(path: Union[Path, str]) -> list[RawPage]:
    """
    Load a plain-text or markdown file as a single page.

    Args:
        path: Path to the .txt or .md file.

    Returns:
        A single-element list so callers treat text files the same as PDFs.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [RawPage(text=text, source=path.name, page_number=1)]
