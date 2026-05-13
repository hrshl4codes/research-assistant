"""
Agno Playground UI — surfaces the research assistant in Agno's interactive
web UI. Run with:

    python playground.py

Then open the URL Agno prints. This complements the CLI; both are valid
interfaces for this project.

Note: The Playground exposes individual agents. Coordinator-routed behavior
is best demonstrated via the CLI (python assistant.py).
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from agents._llm import make_agent
from agents._prompts import RETRIEVER_SYSTEM, GENERAL_SYSTEM
from agents.general import GeneralAgent


def _build_agents():
    """Build Agno Agent instances for Playground registration.

    The general agent is fully functional with calculator and web_search tools.
    For document retrieval, use the CLI: python assistant.py "your question"
    """
    general_agent = make_agent(
        system=GENERAL_SYSTEM,
        tools=_get_general_tools(),
    )
    # The retriever agent in the Playground cannot perform vector search
    # (no DuckDB connection in this context). For grounded doc Q&A, use the CLI.
    retriever_agent = make_agent(
        system=RETRIEVER_SYSTEM + "\n\nNote: In the Playground, document chunks are not pre-loaded. For real retrieval, run: python assistant.py 'your question'",
    )
    return retriever_agent, general_agent


def _get_general_tools():
    """Return the tool functions used by GeneralAgent."""
    from tools.calculator import CalculatorInput, calculate
    from tools.web_search import SearchInput, web_search
    import time
    from schemas.tools import ToolCall

    def calculator_tool(expression: str, context: str = "") -> str:
        """Safely calculate a math expression using sympy."""
        inp = CalculatorInput(expression=expression, context=context)
        out = calculate(inp)
        if not out.success:
            return f"ERROR: {out.error}"
        if out.numeric_value is not None:
            return f"{out.simplified} ~= {out.numeric_value:.10g}"
        return out.simplified

    def web_search_tool(query_str: str, max_results: int = 3) -> str:
        """Search the (mock) web and return the top results."""
        inp = SearchInput(query=query_str, max_results=max_results)
        out = web_search(inp)
        if not out.results:
            return f"No results. ({out.note})"
        lines = [f"({out.source}) {out.note}"]
        for i, r in enumerate(out.results, 1):
            lines.append(f"{i}. {r.title}\n   {r.snippet}")
        return "\n".join(lines)

    return [calculator_tool, web_search_tool]


def main() -> None:
    retriever_agent, general_agent = _build_agents()

    try:
        from agno.playground import Playground
        app = Playground(agents=[retriever_agent, general_agent])
        app.serve(app="playground:app", reload=False)
    except ImportError:
        print(
            "Agno Playground UI not available in this Agno version.\n"
            "The CLI at `python assistant.py` is the primary interface.\n"
            "Run: python assistant.py --repl"
        )
    except Exception as exc:
        print(f"Playground failed to start: {exc}")
        print("Use the CLI: python assistant.py --repl")


if __name__ == "__main__":
    main()
