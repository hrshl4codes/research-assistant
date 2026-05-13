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


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

from rich.console import Console

from agents._llm import make_agent, run_with_fallback
from agents._prompts import RETRIEVER_SYSTEM
from agents._tracing import trace
from rag.embedder import Embedder
from rag.store import search_chunks

console = Console()


class RetrieverAgent:
    """
    Retrieves top-k chunks for a query, then asks the LLM to answer using ONLY
    those chunks. Always returns a RetrievalResult — even on refusal — so the
    UI can show what was retrieved.
    """

    def __init__(self, conn, embedder: Embedder, top_k: int = 4) -> None:
        self._conn = conn
        self._embedder = embedder
        self._top_k = top_k

    @trace("RetrieverAgent.answer")
    def answer(self, query: str) -> RetrievalResult:
        query_vec = self._embedder.encode_single(query)
        results = search_chunks(self._conn, query_vec, top_k=self._top_k)

        if not results:
            return RetrievalResult(
                query=query,
                answer="",
                retrieved_chunks=[],
                confidence="low",
                refusal_reason="The vector store returned no chunks for this query.",
            )

        chunks_block = format_chunks_for_prompt(results)
        prompt = (
            f"User question: {query}\n\n"
            f"Document chunks (use ONLY these to answer):\n{chunks_block}\n\n"
            f"Respond with your answer citing chunk numbers like [chunk 1]. "
            f"Be concise. If the chunks don't address the question, say so plainly."
        )

        def builder(primary: bool):
            return make_agent(
                system=RETRIEVER_SYSTEM,
                primary=primary,
            )

        try:
            raw = run_with_fallback(builder, prompt)
            answer_text = str(raw.content if hasattr(raw, "content") else raw).strip()
        except Exception as exc:
            return RetrievalResult(
                query=query,
                answer="",
                retrieved_chunks=search_results_to_retrieved_chunks(results),
                confidence="low",
                refusal_reason=f"LLM call failed: {exc}",
            )

        if not answer_text:
            return RetrievalResult(
                query=query,
                answer="",
                retrieved_chunks=search_results_to_retrieved_chunks(results),
                confidence="low",
                refusal_reason="The model returned an empty response for this query.",
            )

        confidence = confidence_from_distance(results[0].cosine_distance)

        return RetrievalResult(
            query=query,
            answer=answer_text,
            retrieved_chunks=search_results_to_retrieved_chunks(results),
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Smoke test (requires API key and ingested DB)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()
    from rag.store import get_store

    DB = "data/research_assistant.duckdb"
    if not Path(DB).exists():
        console.print(f"[red]Missing {DB}. Run `python ingest.py` first.[/red]")
        sys.exit(1)

    console.rule("[bold cyan]RetrieverAgent Smoke Test[/bold cyan]")
    embedder = Embedder()
    conn = get_store(DB)
    agent = RetrieverAgent(conn=conn, embedder=embedder, top_k=4)

    queries = [
        "What is this document about?",
        "Summarize the main contribution in one sentence.",
    ]
    for q in queries:
        console.rule(f"[dim]{q}[/dim]")
        result = agent.answer(q)
        console.print(f"[bold]confidence:[/bold] {result.confidence}")
        if result.refusal_reason:
            console.print(f"[yellow]refusal:[/yellow] {result.refusal_reason}")
        console.print(f"[bold]answer:[/bold] {result.answer or '(refused)'}")
        console.print(f"[dim]chunks retrieved: {len(result.retrieved_chunks)}[/dim]\n")
