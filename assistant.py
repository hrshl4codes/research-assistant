"""
Research Assistant — multi-agent CLI.

Usage:
    python assistant.py "your question"
    python assistant.py --agent retriever "your question"     # force retriever
    python assistant.py --agent general "your question"       # force general
    python assistant.py --repl                                # interactive mode
    python assistant.py --verbose "your question"             # show full reasoning
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from rag.embedder import Embedder
from rag.store import get_store
from schemas.results import GeneralResult, RetrievalResult

console = Console()
DB_PATH = "data/research_assistant.duckdb"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _check_db_or_exit() -> None:
    if not Path(DB_PATH).exists():
        console.print(
            f"[red]No knowledge base at {DB_PATH}.[/red]\n"
            f"Run [bold]python ingest.py[/bold] first."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------

def render_decision(decision) -> None:
    """Show the routing decision as a panel — auditability surface."""
    tools = ", ".join(decision.tools_likely_needed) if decision.tools_likely_needed else "none"
    body = (
        f"[bold]Query type:[/bold]     {decision.query_type}\n"
        f"[bold]Selected agent:[/bold] {decision.selected_agent}\n"
        f"[bold]Tools expected:[/bold] {tools}\n"
        f"[bold]Reasoning:[/bold]      {decision.reasoning}"
    )
    console.print(Panel(body, title="[blue]Routing Decision[/blue]", border_style="blue"))


def render_retrieval(result: RetrievalResult, verbose: bool = False) -> None:
    """Pretty-print a RetrievalResult."""
    if result.retrieved_chunks:
        table = Table(title="Retrieved Chunks", show_lines=True, expand=True)
        table.add_column("#", width=3, style="bold")
        table.add_column("Source", width=24, style="dim")
        table.add_column("Text")
        table.add_column("Dist.", justify="right", width=7)
        truncate_at = 500 if verbose else 180
        for i, c in enumerate(result.retrieved_chunks, 1):
            txt = c.text[:truncate_at] + ("..." if len(c.text) > truncate_at else "")
            table.add_row(str(i), c.source, txt, f"{c.cosine_distance:.3f}")
        console.print(table)
    else:
        console.print("[yellow]No chunks retrieved.[/yellow]")

    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}[result.confidence]
    if result.refusal_reason:
        console.print(Panel(
            f"[yellow]Refusal:[/yellow] {result.refusal_reason}",
            title="Answer",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            result.answer,
            title=f"Answer  [{confidence_color}]({result.confidence} confidence)[/{confidence_color}]",
            border_style=confidence_color,
        ))


def render_general(result: GeneralResult, verbose: bool = False) -> None:
    """Pretty-print a GeneralResult."""
    if result.tools_used:
        tbl = Table(title="Tool Calls", show_lines=True, expand=True)
        tbl.add_column("Tool", style="bold")
        tbl.add_column("Input")
        tbl.add_column("OK?", justify="center", width=5)
        tbl.add_column("ms", justify="right", width=8)
        for t in result.tools_used:
            ok = "[green]OK[/green]" if t.success else "[red]X[/red]"
            input_summary = str(t.input)[:80]
            tbl.add_row(t.tool_name, input_summary, ok, f"{t.duration_ms:.1f}")
        console.print(tbl)
    else:
        console.print("[dim]No tools called.[/dim]")

    console.print(Panel(result.answer, title="Answer", border_style="green"))
    if verbose:
        console.print(Panel(result.reasoning_trace, title="Reasoning Trace", border_style="dim"))
    else:
        trimmed = (
            result.reasoning_trace
            if len(result.reasoning_trace) < 300
            else result.reasoning_trace[:300] + "..."
        )
        console.print(f"[dim]Trace: {trimmed}[/dim]")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_once(query: str, agent_name: str, embedder: Embedder, conn, verbose: bool) -> None:
    started = time.perf_counter()

    if agent_name == "retriever":
        from agents.retriever import RetrieverAgent
        result = RetrieverAgent(conn=conn, embedder=embedder).answer(query)
        render_retrieval(result, verbose=verbose)

    elif agent_name == "general":
        from agents.general import GeneralAgent
        result = GeneralAgent().answer(query)
        render_general(result, verbose=verbose)

    else:
        # auto = coordinator
        from agents.coordinator import Coordinator
        coord = Coordinator(conn=conn, embedder=embedder)
        bundle = coord.handle(query)

        render_decision(bundle["decision"])

        if bundle["agent"] == "none":
            console.print(Panel(
                bundle.get("clarification", "Query unclear."),
                title="Clarification needed",
                border_style="yellow",
            ))
        elif bundle["agent"] == "mixed":
            render_retrieval(bundle["retrieval"], verbose=verbose)
            render_general(bundle["general"], verbose=verbose)
        elif bundle["agent"] == "retriever_agent":
            render_retrieval(bundle["result"], verbose=verbose)
        else:
            render_general(bundle["result"], verbose=verbose)

    elapsed = time.perf_counter() - started
    console.print(f"[dim]Total: {elapsed:.2f}s[/dim]\n")


def run_repl(agent_name: str, embedder: Embedder, conn, verbose: bool) -> None:
    console.print("[dim]Interactive mode. Type 'exit' or Ctrl-D to quit.[/dim]\n")
    while True:
        try:
            query = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return
        console.print(Panel(query, title="[bold white]Query[/bold white]", border_style="white"))
        run_once(query, agent_name, embedder, conn, verbose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold cyan]Research Assistant[/bold cyan]", style="cyan")

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", help="The question to ask")
    p.add_argument("--agent", choices=["auto", "retriever", "general"], default="auto")
    p.add_argument("--repl", action="store_true", help="Start interactive REPL")
    p.add_argument("--verbose", action="store_true", help="Show full reasoning traces")
    args = p.parse_args()

    if not args.repl and not args.query:
        p.error("must provide a query or use --repl")

    _check_db_or_exit()

    embedder = Embedder()
    conn = get_store(DB_PATH)

    if args.repl:
        run_repl(args.agent, embedder, conn, args.verbose)
    else:
        console.print(Panel(
            args.query,
            title="[bold white]Query[/bold white]",
            border_style="white",
        ))
        run_once(args.query, args.agent, embedder, conn, args.verbose)


if __name__ == "__main__":
    main()
