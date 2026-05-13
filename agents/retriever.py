"""
Retriever agent — answers queries that require grounded document context.

Responsibilities:
- Embed the user query and retrieve top-k chunks from the DuckDB VSS store.
- Expose the retrieved chunks explicitly in the response (not silently consumed).
- Generate an answer that cites which chunk(s) it drew from.
"""

from __future__ import annotations

from typing import Optional

from rag.store import SearchResult
from schemas.results import Confidence, RetrievedChunk, RetrievalResult

_HIGH_MAX = 0.30
_MEDIUM_MAX = 0.50


def format_chunks_for_prompt(results: list[SearchResult]) -> str:
    """Render retrieved chunks as a numbered, citation-friendly block for the LLM."""
    if not results:
        return "[no chunks retrieved]"
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[chunk {i}] source={r.source} distance={r.cosine_distance:.3f}\n"
            f"{r.text}\n"
        )
    return "\n".join(lines)


def confidence_from_distance(top_distance: Optional[float]) -> Confidence:
    """Map the top retrieved chunk's cosine distance to a confidence label."""
    if top_distance is None:
        return "low"
    if top_distance < _HIGH_MAX:
        return "high"
    if top_distance < _MEDIUM_MAX:
        return "medium"
    return "low"


def search_results_to_retrieved_chunks(results: list[SearchResult]) -> list[RetrievedChunk]:
    """Convert internal SearchResult objects into the schema's presentation type."""
    return [
        RetrievedChunk(
            text=r.text,
            source=r.source,
            cosine_distance=r.cosine_distance,
            doc_id=r.doc_id,
        )
        for r in results
    ]
