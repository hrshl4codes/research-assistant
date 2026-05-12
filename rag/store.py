"""
DuckDB vector store backed by the vss extension.

Responsibilities:
- Install and load the DuckDB vss extension (provides HNSW index + array distance functions).
- Manage the `chunks` table that holds embedded document fragments.
- Expose insert and similarity-search operations with typed Pydantic I/O.

Design note: the HNSW index is created in a separate call (`create_hnsw_index`)
rather than automatically on table creation.  DuckDB's HNSW index is built over
existing rows; calling it before any data exists still works but yields a zero-
node graph.  Callers should call `create_hnsw_index` once after the initial bulk
load so the index is warm before query time.  Searches fall back to a full table
scan when no index exists, so correctness is never affected, only latency.
"""

from __future__ import annotations

import logging
from typing import Union

import duckdb
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A single embedded document fragment ready for insertion."""

    id: str = Field(..., description="Stable unique identifier for this chunk.")
    doc_id: str = Field(..., description="Identifier of the parent document.")
    text: str = Field(..., description="Raw text of the chunk.")
    source: str = Field(..., description="Human-readable source label (e.g. filename).")
    embedding: list[float] = Field(
        ...,
        description="384-dimensional float embedding produced by all-MiniLM-L6-v2.",
    )


class SearchResult(BaseModel):
    """A retrieved chunk together with its distance from the query vector."""

    id: str
    doc_id: str
    text: str
    source: str
    cosine_distance: float = Field(
        ...,
        description=(
            "Cosine distance in [0, 2].  Lower is more similar.  "
            "0 = identical direction; 2 = opposite direction."
        ),
    )


# ---------------------------------------------------------------------------
# Store lifecycle
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 384

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id      VARCHAR PRIMARY KEY,
    doc_id  VARCHAR  NOT NULL,
    text    TEXT     NOT NULL,
    source  VARCHAR  NOT NULL,
    embedding FLOAT[{_EMBEDDING_DIM}] NOT NULL
)
"""

_CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS embedding_hnsw
ON chunks
USING HNSW (embedding)
WITH (metric = 'cosine')
"""


def get_store(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Open (or create) a DuckDB database, enable the vss extension, and
    ensure the `chunks` table exists.

    Args:
        db_path: Filesystem path for a persistent database, or ':memory:'
                 for a transient in-process store.

    Returns:
        An open DuckDB connection ready for insert / search operations.

    Raises:
        RuntimeError: If the vss extension cannot be installed or loaded.
    """
    conn = duckdb.connect(db_path)

    try:
        conn.execute("INSTALL vss;")
        conn.execute("LOAD vss;")
    except Exception as exc:
        raise RuntimeError(
            "Failed to install/load DuckDB vss extension. "
            "Ensure you have an internet connection for the first install, "
            f"and that duckdb>=0.10.0 is installed. Original error: {exc}"
        ) from exc

    # HNSW persistence is experimental in DuckDB 1.x and must be opted in
    # explicitly for file-backed databases. In-memory databases don't need it.
    if db_path != ":memory:":
        conn.execute("SET hnsw_enable_experimental_persistence = true;")

    conn.execute(_CREATE_TABLE_SQL)
    logger.debug("DuckDB store ready at '%s'.", db_path)
    return conn


def create_hnsw_index(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Build an HNSW approximate-nearest-neighbour index on the embedding column.

    Call this once after the initial bulk load.  Subsequent inserts update the
    index automatically.  Safe to call repeatedly — uses IF NOT EXISTS.

    Args:
        conn: An open connection returned by `get_store`.
    """
    conn.execute(_CREATE_INDEX_SQL)
    logger.info("HNSW index created (or already existed).")


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


def insert_chunk(
    conn: duckdb.DuckDBPyConnection,
    chunk: Chunk,
) -> None:
    """
    Insert a single chunk into the store.

    Args:
        conn:  An open connection returned by `get_store`.
        chunk: Validated Chunk instance.  The embedding must have exactly
               384 elements.

    Raises:
        ValueError: If the embedding has the wrong dimensionality.
    """
    if len(chunk.embedding) != _EMBEDDING_DIM:
        raise ValueError(
            f"Embedding has {len(chunk.embedding)} dimensions; "
            f"expected {_EMBEDDING_DIM}."
        )

    conn.execute(
        """
        INSERT OR REPLACE INTO chunks (id, doc_id, text, source, embedding)
        VALUES (?, ?, ?, ?, ?)
        """,
        [chunk.id, chunk.doc_id, chunk.text, chunk.source, chunk.embedding],
    )


# ---------------------------------------------------------------------------
# Bulk insert
# ---------------------------------------------------------------------------


def insert_chunks_batch(
    conn: duckdb.DuckDBPyConnection,
    chunks: list[Chunk],
) -> None:
    """
    Insert multiple chunks in a single transaction.

    One transaction is 10-100x faster than individual auto-committed inserts for
    large batches: DuckDB flushes its write-ahead log once per transaction, not
    once per row.

    Uses INSERT OR REPLACE, so re-ingesting the same document is idempotent.
    Existing rows with the same id are overwritten.

    Args:
        conn:   An open connection returned by `get_store`.
        chunks: Validated Chunk instances to insert. Empty list is a no-op.

    Raises:
        ValueError: If any chunk has an embedding with the wrong dimensionality.
    """
    if not chunks:
        return

    for chunk in chunks:
        if len(chunk.embedding) != _EMBEDDING_DIM:
            raise ValueError(
                f"Chunk '{chunk.id}' has {len(chunk.embedding)}-dim embedding; "
                f"expected {_EMBEDDING_DIM}."
            )

    records = [
        [c.id, c.doc_id, c.text, c.source, c.embedding]
        for c in chunks
    ]

    conn.execute("BEGIN TRANSACTION;")
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (id, doc_id, text, source, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise

    logger.debug("Batch-inserted %d chunk(s).", len(chunks))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_chunks(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: Union[list[float], "np.ndarray"],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Return the top-k chunks closest to `query_embedding` by cosine distance.

    Uses the HNSW index when available (approximate but fast).  If the index
    has not been created yet, DuckDB falls back to a full table scan
    (exact but O(n)).

    Args:
        conn:            An open connection returned by `get_store`.
        query_embedding: 384-element query vector.  Accepts a Python list or
                         a numpy array — both are coerced to list[float].
        top_k:           Number of nearest neighbours to return.

    Returns:
        List of SearchResult objects sorted by ascending cosine_distance
        (most similar first).
    """
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.astype(np.float32).tolist()

    if len(query_embedding) != _EMBEDDING_DIM:
        raise ValueError(
            f"Query embedding has {len(query_embedding)} dimensions; "
            f"expected {_EMBEDDING_DIM}."
        )

    rows = conn.execute(
        f"""
        SELECT
            id,
            doc_id,
            text,
            source,
            array_cosine_distance(embedding, ?::FLOAT[{_EMBEDDING_DIM}]) AS cosine_distance
        FROM chunks
        ORDER BY cosine_distance ASC
        LIMIT ?
        """,
        [query_embedding, top_k],
    ).fetchall()

    return [
        SearchResult(
            id=row[0],
            doc_id=row[1],
            text=row[2],
            source=row[3],
            cosine_distance=float(row[4]),
        )
        for row in rows
    ]


def count_chunks(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the total number of chunks currently in the store."""
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid

    from rich.console import Console
    from rich.table import Table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    console = Console()

    console.rule("[bold cyan]DuckDB VSS Smoke Test[/bold cyan]")

    # 1. Open in-memory store and load vss.
    store = get_store(":memory:")
    console.print("[green]vss extension loaded.[/green]")

    # 2. Insert 3 fake chunks with random 384-dim embeddings.
    rng = np.random.default_rng(seed=42)
    fake_chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            doc_id=f"doc_{i}",
            text=f"This is fake document chunk number {i}. "
                 f"It contains placeholder text for testing.",
            source=f"fake_doc_{i}.pdf",
            embedding=rng.random(384).astype(np.float32).tolist(),
        )
        for i in range(3)
    ]

    for chunk in fake_chunks:
        insert_chunk(store, chunk)

    console.print(f"[green]Inserted {count_chunks(store)} chunks.[/green]")

    # 3. Build the HNSW index.
    create_hnsw_index(store)
    console.print("[green]HNSW index built.[/green]")

    # 4. Run a similarity search with a fresh random query vector.
    query_vec = rng.random(384).astype(np.float32)
    results = search_chunks(store, query_vec, top_k=3)

    # 5. Display results.
    table = Table(title="Top-3 Similar Chunks", show_lines=True)
    table.add_column("Rank", style="bold", width=6)
    table.add_column("ID", style="dim", width=38)
    table.add_column("Doc", width=8)
    table.add_column("Text (truncated)", width=50)
    table.add_column("Cosine Dist.", justify="right", width=14)

    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            result.id,
            result.doc_id,
            result.text[:60] + "…",
            f"{result.cosine_distance:.6f}",
        )

    console.print(table)
    console.print(
        f"\n[bold green]VSS smoke test PASSED.[/bold green] "
        f"Returned {len(results)} result(s) from {count_chunks(store)} chunk(s)."
    )
