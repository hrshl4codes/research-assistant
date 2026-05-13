"""
Assessment runner. Loads assessments/dataset.json, runs each query through the
coordinator, scores routing accuracy and tool selection, writes a timestamped
report to logs/assessment-<timestamp>.json.

Usage:
    python -m assessments.run                # full run (makes LLM calls)
    python -m assessments.run --quick        # routing-only, no agent execution
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agents.coordinator import Coordinator
from rag.embedder import Embedder
from rag.store import get_store

console = Console()
DATASET = Path(__file__).parent / "dataset.json"
DB = "data/research_assistant.duckdb"


def grade_agent(expected: str, actual: str) -> bool:
    """`expected` is one of: retriever_agent | general_agent | mixed | none."""
    return expected == actual


def grade_tools(expected: list[str], actual: list[str]) -> bool:
    """Pass if expected tools are a subset of actual tools called."""
    return set(expected).issubset(set(actual))


def run() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Routing-only — skip agent execution")
    args = p.parse_args()

    cases = json.loads(DATASET.read_text())
    embedder = Embedder()
    conn = get_store(DB)
    coord = Coordinator(conn=conn, embedder=embedder)

    rows = []
    routing_correct = 0
    tools_correct = 0

    for case in cases:
        q = case["query"]
        console.rule(f"[dim]{case['id']}: {q[:60]}[/dim]")
        started = time.perf_counter()

        if args.quick:
            decision = coord.route(q)
            actual_agent = {
                "document_qa": "retriever_agent",
                "general_knowledge": "general_agent",
                "calculation": "general_agent",
                "web_search": "general_agent",
                "mixed": "mixed",
                "unclear": "none",
            }.get(decision.query_type, decision.selected_agent)
            actual_tools: list[str] = list(decision.tools_likely_needed)
        else:
            bundle = coord.handle(q)
            actual_agent = bundle["agent"]
            actual_tools = []
            if "result" in bundle and bundle["result"] and hasattr(bundle["result"], "tools_used"):
                actual_tools = [t.tool_name for t in bundle["result"].tools_used]
            if "general" in bundle and hasattr(bundle.get("general"), "tools_used"):
                actual_tools += [t.tool_name for t in bundle["general"].tools_used]

        elapsed = time.perf_counter() - started
        agent_ok = grade_agent(case["expected_agent"], actual_agent)
        tools_ok = grade_tools(case["expected_tools"], actual_tools)
        routing_correct += int(agent_ok)
        tools_correct += int(tools_ok)
        rows.append({
            "id": case["id"],
            "query": q,
            "expected_agent": case["expected_agent"],
            "actual_agent": actual_agent,
            "agent_ok": agent_ok,
            "expected_tools": case["expected_tools"],
            "actual_tools": actual_tools,
            "tools_ok": tools_ok,
            "elapsed_s": round(elapsed, 2),
        })
        status_color = "green" if agent_ok else "red"
        console.print(f"  agent: {actual_agent} [{status_color}]({'OK' if agent_ok else 'MISS'})[/{status_color}]")
        if case["expected_tools"]:
            tool_color = "green" if tools_ok else "red"
            console.print(f"  tools: {actual_tools} [{tool_color}]({'OK' if tools_ok else 'MISS'})[/{tool_color}]")

    n = len(cases)
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "quick": args.quick,
        "n": n,
        "routing_accuracy": routing_correct / n,
        "tool_accuracy": tools_correct / n,
        "cases": rows,
    }

    Path("logs").mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(f"logs/assessment-{ts}.json")
    out.write_text(json.dumps(summary, indent=2, default=str))

    table = Table(title=f"Assessment Summary — {ts}")
    table.add_column("Case", style="bold")
    table.add_column("Agent")
    table.add_column("Tools")
    table.add_column("ms", justify="right")
    for r in rows:
        table.add_row(
            r["id"],
            f"[{'green' if r['agent_ok'] else 'red'}]{r['actual_agent']}[/]",
            f"[{'green' if r['tools_ok'] else 'red'}]{','.join(r['actual_tools']) or '-'}[/]",
            f"{int(r['elapsed_s'] * 1000)}",
        )
    console.print(table)
    console.print(
        f"\n[bold]Routing accuracy:[/bold] {routing_correct}/{n} = {routing_correct / n * 100:.0f}%"
    )
    console.print(
        f"[bold]Tool-selection accuracy:[/bold] {tools_correct}/{n} = {tools_correct / n * 100:.0f}%"
    )
    console.print(f"[dim]Report saved to {out}[/dim]")


if __name__ == "__main__":
    run()
