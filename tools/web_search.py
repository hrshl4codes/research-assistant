"""
Web search tool. Mocked, but transparent about it.

Returns a structured SearchOutput that labels results as "mock_data" so
the agent can cite the source honestly. The mock covers 5 representative
queries with plausible results. Anything else gets an empty result list.

To wire up a real search API, replace the body of web_search() while
keeping the input/output schemas identical.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    """What the agent sends to the web search tool."""

    query: str = Field(..., description="Search query string.")
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of results to return.",
    )


class SearchResultItem(BaseModel):
    """A single search result."""

    title: str
    snippet: str
    url: str


class SearchOutput(BaseModel):
    """What the web search tool returns to the agent."""

    query: str = Field(..., description="Original query, echoed back.")
    results: list[SearchResultItem]
    result_count: int = Field(..., description="Number of results returned.")
    source: Literal["mock_data", "live"] = Field(
        ...,
        description=(
            "'mock_data' for hardcoded results; "
            "'live' once a real search API is wired up."
        ),
    )
    note: str = Field(
        default="",
        description="Human-readable note explaining the result (e.g. mock disclaimer).",
    )


# ---------------------------------------------------------------------------
# Mock database — 5 representative queries
# ---------------------------------------------------------------------------
# Keys are lowercase normalized queries. Values are lists of result dicts
# that get converted to SearchResultItem on the way out.

_MOCK_DB: dict[str, list[dict]] = {
    "what is retrieval augmented generation": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "snippet": (
                "RAG combines a dense retriever (DPR) with a seq2seq generator (BART). "
                "The retriever fetches relevant passages from Wikipedia; the generator "
                "conditions on them to produce answers. Outperforms parametric-only "
                "models on open-domain QA benchmarks."
            ),
            "url": "https://arxiv.org/abs/2005.11401",
        },
        {
            "title": "RAG explained: grounding LLMs in external knowledge",
            "snippet": (
                "RAG reduces hallucination by injecting retrieved documents into the "
                "prompt before generation. The key trade-off: retrieval adds latency "
                "but dramatically improves factual accuracy on domain-specific queries."
            ),
            "url": "https://research.ibm.com/blog/retrieval-augmented-generation-RAG",
        },
        {
            "title": "Practical RAG: chunking, embedding, and retrieval strategies",
            "snippet": (
                "Chunk size, overlap, and embedding model choice all affect retrieval "
                "quality. Smaller chunks (256–512 chars) improve precision; larger "
                "chunks (1024+) preserve more context. Hybrid sparse+dense retrieval "
                "often outperforms either alone."
            ),
            "url": "https://www.pinecone.io/learn/retrieval-augmented-generation/",
        },
    ],
    "rag versus fine-tuning for llms": [
        {
            "title": "RAG vs. fine-tuning: when to use which",
            "snippet": (
                "Fine-tuning updates model weights permanently and is best for teaching "
                "a new task format or style. RAG leaves weights unchanged and is better "
                "for injecting up-to-date or proprietary knowledge. Combining both is "
                "common in production systems."
            ),
            "url": "https://www.anyscale.com/blog/fine-tuning-vs-rag",
        },
        {
            "title": "Fine-tuning is not enough: the case for RAG in enterprise AI",
            "snippet": (
                "Fine-tuned models still hallucinate when queried outside their training "
                "distribution. RAG provides a verifiable evidence trail — the retrieved "
                "chunks can be shown to the user, making the system auditable."
            ),
            "url": "https://hai.stanford.edu/news/rag-enterprise-knowledge",
        },
    ],
    "how does hnsw indexing work": [
        {
            "title": "Efficient and robust approximate nearest neighbor search using HNSW",
            "snippet": (
                "HNSW builds a multi-layer graph where higher layers are coarse "
                "skip-graphs and lower layers are dense proximity graphs. Query time "
                "is O(log n) for construction and sub-linear for search. Outperforms "
                "IVF-Flat on recall-vs-latency tradeoff."
            ),
            "url": "https://arxiv.org/abs/1603.09320",
        },
        {
            "title": "DuckDB VSS: HNSW vectors in an analytical database",
            "snippet": (
                "The duckdb-vss extension adds HNSW indexing to DuckDB columns of type "
                "FLOAT[N]. Cosine, L2, and inner-product metrics are supported. "
                "Experimental persistence support is available via the "
                "hnsw_enable_experimental_persistence setting."
            ),
            "url": "https://duckdb.org/docs/extensions/vss.html",
        },
    ],
    "federated learning and data privacy": [
        {
            "title": "Communication-efficient learning of deep networks from decentralized data",
            "snippet": (
                "The FedAvg paper introduced the foundational federated learning algorithm. "
                "Clients train locally for multiple epochs, then send weight updates to a "
                "server that averages them. Reduces communication rounds by 10-100x vs "
                "naive gradient sharing."
            ),
            "url": "https://arxiv.org/abs/1602.05629",
        },
        {
            "title": "Differential privacy in federated learning",
            "snippet": (
                "Even with federated training, gradient updates can leak training data "
                "through model inversion attacks. Differential privacy (DP) adds calibrated "
                "Gaussian noise to gradients before transmission, providing a formal "
                "privacy guarantee at the cost of some model accuracy."
            ),
            "url": "https://blog.research.google/2017/04/federated-learning-collaborative.html",
        },
    ],
    "transformer self-attention mechanism": [
        {
            "title": "Attention Is All You Need",
            "snippet": (
                "Vaswani et al. (2017) replaced recurrence with multi-head self-attention. "
                "Each attention head computes Q, K, V projections and scores tokens by "
                "softmax(QK^T / sqrt(d_k))V. Parallelises fully over sequence length, "
                "enabling much faster training than RNNs."
            ),
            "url": "https://arxiv.org/abs/1706.03762",
        },
        {
            "title": "The illustrated transformer",
            "snippet": (
                "A visual walkthrough of self-attention, positional encoding, and the "
                "encoder-decoder stack. Widely cited as the clearest introduction to "
                "transformer internals for practitioners."
            ),
            "url": "https://jalammar.github.io/illustrated-transformer/",
        },
    ],
}

_NOTE_MOCK = (
    "Results are from a hardcoded mock database. "
    "Wire up a real search API to get live results."
)
_NOTE_NO_RESULTS = (
    "No mock results for this query. "
    "Try one of the 5 pre-loaded queries or connect a live search API."
)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


def web_search(inp: SearchInput) -> SearchOutput:
    """
    Look up a query in the mock database and return matching results.

    Matching is case-insensitive and strips leading/trailing whitespace.
    If the query doesn't match any of the 5 pre-loaded entries, an empty
    result list is returned with source='mock_data' so the agent knows
    to say it couldn't find anything rather than hallucinating.

    Args:
        inp: SearchInput with query and max_results.

    Returns:
        SearchOutput with results, result count, source label, and a note.
    """
    normalized = inp.query.strip().lower()
    raw_results = _MOCK_DB.get(normalized, [])
    clipped = raw_results[: inp.max_results]

    items = [SearchResultItem(**r) for r in clipped]

    return SearchOutput(
        query=inp.query,
        results=items,
        result_count=len(items),
        source="mock_data",
        note=_NOTE_MOCK if items else _NOTE_NO_RESULTS,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold cyan]Web Search Tool — Self-Test[/bold cyan]")

    tests = [
        SearchInput(query="what is retrieval augmented generation", max_results=2),
        SearchInput(query="how does hnsw indexing work", max_results=2),
        SearchInput(query="latest Premier League results"),  # no mock data
    ]

    for t in tests:
        out = web_search(t)
        console.print(
            f"\n[bold]Query:[/bold] {out.query}  "
            f"[dim]source={out.source}  count={out.result_count}[/dim]"
        )
        console.print(f"[dim italic]{out.note}[/dim italic]")

        if out.results:
            table = Table(show_lines=True, expand=True)
            table.add_column("Title", width=40)
            table.add_column("Snippet")
            table.add_column("URL", width=30, style="dim")
            for r in out.results:
                table.add_row(r.title, r.snippet[:120] + "…", r.url)
            console.print(table)
        else:
            console.print("[yellow]  No results.[/yellow]")
