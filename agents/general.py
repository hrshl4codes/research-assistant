"""
General agent — answers queries that don't require document retrieval.

Responsibilities:
- Handle open-domain questions using the LLM's parametric knowledge.
- Invoke the calculator tool for arithmetic/symbolic math.
- Invoke the web_search tool for current-events lookups.

Tool calls are recorded via a per-call audit list and injected into
GeneralResult.tools_used after the LLM run completes. This is necessary
because Agno 1.4.4 does not expose a stable structured tool-call log.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from agents._llm import make_agent, run_with_fallback
from agents._prompts import GENERAL_SYSTEM
from schemas.results import GeneralResult
from schemas.tools import ToolCall
from tools.calculator import CalculatorInput, calculate
from tools.web_search import SearchInput, web_search

console = Console()


class GeneralAgent:
    """LLM agent with calculator and web_search tools."""

    def __init__(self) -> None:
        pass

    def answer(self, query: str) -> GeneralResult:
        # Per-call audit log — cleared at the start of every answer() call.
        call_log: list[ToolCall] = []

        # ------------------------------------------------------------------ #
        # Tool wrappers — record into call_log, return strings to the LLM    #
        # ------------------------------------------------------------------ #

        def calculator_tool(expression: str, context: str = "") -> str:
            """
            Safely calculate a math expression using sympy.

            Args:
                expression: The math expression, e.g. "2**10 + sqrt(16)".
                context: Optional human-readable note about why this calculation is happening.

            Returns:
                A short string with the simplified form and (when finite) the numeric value.
            """
            inp = CalculatorInput(expression=expression, context=context)
            t0 = time.perf_counter()
            out = calculate(inp)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            call_log.append(
                ToolCall(
                    tool_name="calculator",
                    input=inp.model_dump(),
                    output=out.model_dump(),
                    duration_ms=duration_ms,
                    success=out.success,
                )
            )
            if not out.success:
                return f"ERROR: {out.error}"
            if out.numeric_value is not None:
                return f"{out.simplified} ~= {out.numeric_value:.10g}"
            return out.simplified

        def web_search_tool(query_str: str, max_results: int = 3) -> str:
            """
            Search the (mock) web for a query and return the top results.

            Args:
                query_str: The search query string.
                max_results: How many results to return (1-10).

            Returns:
                A formatted string listing the results, or a note explaining no results.
            """
            inp = SearchInput(query=query_str, max_results=max_results)
            t0 = time.perf_counter()
            out = web_search(inp)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            call_log.append(
                ToolCall(
                    tool_name="web_search",
                    input=inp.model_dump(),
                    output=out.model_dump(),
                    duration_ms=duration_ms,
                    success=out.result_count > 0,
                )
            )
            if not out.results:
                return f"No results. ({out.note})"
            lines = [f"({out.source}) {out.note}"]
            for i, r in enumerate(out.results, 1):
                lines.append(f"{i}. {r.title}\n   {r.snippet}")
            return "\n".join(lines)

        # ------------------------------------------------------------------ #
        # Run the agent                                                        #
        # ------------------------------------------------------------------ #

        def builder(primary: bool):
            return make_agent(
                system=GENERAL_SYSTEM,
                tools=[calculator_tool, web_search_tool],
                primary=primary,
            )

        prompt = (
            f"User query: {query}\n\n"
            f"Decide if you need calculator_tool or web_search_tool. If not, "
            f"answer from your own knowledge. Give your full answer and explain "
            f"your reasoning step by step."
        )

        try:
            raw = run_with_fallback(builder, prompt)
            answer_text = str(raw.content if hasattr(raw, "content") else raw).strip()
            reasoning = f"Query processed via LLM. Tools called: {[t.tool_name for t in call_log] or 'none'}."
        except Exception as exc:
            return GeneralResult(
                query=query,
                answer="",
                tools_used=list(call_log),
                reasoning_trace=f"LLM call failed: {exc}",
            )

        return GeneralResult(
            query=query,
            answer=answer_text,
            tools_used=list(call_log),
            reasoning_trace=reasoning,
        )


# ---------------------------------------------------------------------------
# Smoke test (requires API key)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    console.rule("[bold cyan]GeneralAgent Smoke Test[/bold cyan]")
    agent = GeneralAgent()

    queries = [
        "What is 2**10 + sqrt(16)?",
        "Why is the sky blue?",
    ]
    for q in queries:
        console.rule(f"[dim]{q}[/dim]")
        result = agent.answer(q)
        console.print(f"[bold]answer:[/bold] {result.answer}\n")
        console.print(f"[bold]tools used:[/bold] {[t.tool_name for t in result.tools_used]}")
        console.print(f"[bold]reasoning:[/bold] {result.reasoning_trace}\n")
