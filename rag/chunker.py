"""
Text chunker. Splits raw document pages into overlapping fragments.

Uses LangChain's RecursiveCharacterTextSplitter, which prefers paragraph
boundaries, then sentence boundaries, then word boundaries, never splitting
mid-word. The 150-char overlap means a sentence that straddles a boundary
appears in full in at least one chunk.

Pages are chunked independently so no chunk ever spans two pages. You always
know which page a chunk came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import RawPage

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class TextChunk:
    """A document fragment ready for embedding and storage."""

    text: str
    source: str       # citation string, e.g. "paper.pdf, page 3"
    doc_id: str       # the filename, used as document identifier in the store
    page_number: int
    chunk_index: int  # global sequential index across the whole document


def chunk_pages(
    pages: list[RawPage],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split a list of RawPages into TextChunks, preserving page metadata on each.

    Args:
        pages:         Output of load_pdf() or load_text().
        chunk_size:    Maximum characters per chunk (default 1000).
        chunk_overlap: Overlap between consecutive chunks (default 150).

    Returns:
        Flat list of TextChunk objects in document order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )

    chunks: list[TextChunk] = []
    global_idx = 0

    for page in pages:
        splits = splitter.split_text(page.text)
        for split in splits:
            stripped = split.strip()
            if not stripped:
                continue
            chunks.append(
                TextChunk(
                    text=stripped,
                    source=f"{page.source}, page {page.page_number}",
                    doc_id=page.source,
                    page_number=page.page_number,
                    chunk_index=global_idx,
                )
            )
            global_idx += 1

    logger.info(
        "Chunked %d page(s) → %d chunks (size=%d, overlap=%d).",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
