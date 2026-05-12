"""
Ingestion script. Runs the full RAG pipeline on a PDF document.

Pipeline:  PDF → pages (PyMuPDF) → chunks (LangChain splitter)
           → embeddings (all-MiniLM-L6-v2) → DuckDB VSS store

Usage:
    python ingest.py [PDF_PATH] [--db DB_PATH] [--query QUERY] [--top-k N]

Defaults:
    PDF_PATH : data/sample.pdf
    DB_PATH  : data/research_assistant.duckdb
    QUERY    : "What is this document about?"
    TOP_K    : 3
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.chunker import TextChunk, chunk_pages
from rag.embedder import Embedder
from rag.loader import load_pdf
from rag.store import (
    Chunk,
    count_chunks,
    create_hnsw_index,
    get_store,
    insert_chunks_batch,
    search_chunks,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")
console = Console()


def _chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic ID based on content. Re-ingesting the same document produces
    the same IDs, which makes INSERT OR REPLACE idempotent.
    """
    raw = f"{doc_id}::{chunk_index}::{text[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _build_store_chunks(
    text_chunks: list[TextChunk],
    embeddings: "np.ndarray",  # type: ignore[name-defined]
) -> list[Chunk]:
    return [
        Chunk(
            id=_chunk_id(tc.doc_id, tc.chunk_index, tc.text),
            doc_id=tc.doc_id,
            text=tc.text,
            source=tc.source,
            embedding=embeddings[i].tolist(),
        )
        for i, tc in enumerate(text_chunks)
    ]


def run_pipeline(
    pdf_path: Path,
    db_path: str,
    embedder: Embedder,
) -> dict:
    """
    Execute the full load → chunk → embed → store pipeline.

    Args:
        pdf_path: Path to the PDF to ingest.
        db_path:  DuckDB database path (file or ':memory:').
        embedder: Pre-loaded Embedder instance (avoids reloading the model).

    Returns:
        Dict of timing/count stats plus the open DuckDB connection under key 'conn'.
    """
    t_start = time.perf_counter()

    console.print(f"\n[bold cyan]Step 1/4[/bold cyan]  Loading [italic]{pdf_path.name}[/italic] …")
    pages = load_pdf(pdf_path)
    console.print(f"  → {len(pages)} non-blank page(s)")

    console.print("[bold cyan]Step 2/4[/bold cyan]  Chunking …")
    text_chunks = chunk_pages(pages)
    console.print(f"  → {len(text_chunks)} chunk(s)  (size=1000, overlap=150)")

    console.print(f"[bold cyan]Step 3/4[/bold cyan]  Embedding {len(text_chunks)} chunk(s) …")
    t_embed = time.perf_counter()
    embeddings = embedder.encode(
        [tc.text for tc in text_chunks],
        show_progress=len(text_chunks) > 30,
    )
    embed_time = time.perf_counter() - t_embed
    console.print(f"  → done in {embed_time:.2f}s")

    console.print("[bold cyan]Step 4/4[/bold cyan]  Storing in DuckDB …")
    store_chunks = _build_store_chunks(text_chunks, embeddings)
    conn = get_store(db_path)
    insert_chunks_batch(conn, store_chunks)
    create_hnsw_index(conn)
    console.print(f"  → {count_chunks(conn)} total chunk(s) in store")

    return {
        "pages": len(pages),
        "chunks": len(store_chunks),
        "total_in_store": count_chunks(conn),
        "embed_time": embed_time,
        "total_time": time.perf_counter() - t_start,
        "conn": conn,
    }


def _print_stats(stats: dict) -> None:
    table = Table(title="Ingestion Stats", show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Pages loaded", str(stats["pages"]))
    table.add_row("Chunks created", str(stats["chunks"]))
    table.add_row("Total chunks in store", str(stats["total_in_store"]))
    table.add_row("Embedding time", f"{stats['embed_time']:.2f}s")
    table.add_row("Total pipeline time", f"{stats['total_time']:.2f}s")
    console.print(table)


def _print_search_results(conn, embedder: Embedder, query: str, top_k: int) -> None:
    """Run a vector search and display the top chunks with source citations."""
    console.rule("[bold cyan]Vector Search Results[/bold cyan]")
    console.print(f"[bold]Query:[/bold] {query}\n")

    query_vec = embedder.encode_single(query)
    results = search_chunks(conn, query_vec, top_k=top_k)

    table = Table(show_lines=True, expand=True)
    table.add_column("#", width=3, style="bold")
    table.add_column("Source", width=24, style="dim")
    table.add_column("Retrieved chunk text")
    table.add_column("Dist.", width=7, justify="right")

    for rank, r in enumerate(results, 1):
        table.add_row(
            str(rank),
            r.source,
            r.text[:300] + ("…" if len(r.text) > 300 else ""),
            f"{r.cosine_distance:.4f}",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the research assistant knowledge base."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="data/sample.pdf",
        help="Path to PDF (default: data/sample.pdf)",
    )
    parser.add_argument(
        "--db",
        default="data/research_assistant.duckdb",
        help="DuckDB database path",
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Test query for the vector search demo",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to return")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] '{pdf_path}' not found.\n"
            "Drop a PDF into data/ and pass its path as the first argument."
        )
        sys.exit(1)

    console.rule("[bold cyan]RAG Ingestion Pipeline[/bold cyan]")

    # Load embedding model once — reuse for chunk encoding and query encoding.
    console.print("[dim]Loading embedding model (first run downloads ~80 MB) …[/dim]")
    embedder = Embedder()

    stats = run_pipeline(pdf_path, args.db, embedder)
    conn = stats.pop("conn")

    console.rule("[bold cyan]Stats[/bold cyan]")
    _print_stats(stats)

    _print_search_results(conn, embedder, args.query, args.top_k)

    console.print(
        "\n[bold green]Ingestion complete.[/bold green]  "
        "RAG pipeline ready for the next stage.\n"
    )


if __name__ == "__main__":
    main()
