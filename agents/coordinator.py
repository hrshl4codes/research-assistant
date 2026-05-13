"""
Coordinator agent — routes incoming queries to the correct specialist agent.

Responsibilities:
- Classify query intent (document-grounded vs. general knowledge vs. tool-required).
- Emit a structured RoutingDecision with explicit reasoning so decisions are auditable.
- Delegate to RetrieverAgent or GeneralAgent accordingly.
"""

from __future__ import annotations

import re

# Deterministic arithmetic pre-filter — matches queries that are mostly math
# operators and numbers, so we can route to the calculator without an LLM call.

_MATH_TOKENS = re.compile(r"[+\-*/^]|\*\*|sqrt|sin|cos|tan|log|ln|exp|pi|\be\b")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
_CALC_WORDS = re.compile(r"\b(compute|calculate|what\s*is)\b", re.IGNORECASE)


def looks_like_arithmetic(query: str) -> bool:
    """
    True if the query looks like pure arithmetic (likely candidate for calculator).

    Heuristic — favours specificity over recall. False negatives are fine; the
    LLM router will handle them. False positives are bounded by the calculator's
    own error handling.
    """
    q = query.lower().strip()
    has_numbers = len(_NUMBER.findall(q)) >= 1
    has_math_tokens = bool(_MATH_TOKENS.search(q))
    has_calc_word = bool(_CALC_WORDS.search(q))

    if len(q) > 80:
        return False

    if re.search(r"\b(explain|describe|compare|summari[sz]e|why|how does|tell me about|what is the (document|paper)|who)\b", q):
        return False

    return has_numbers and (has_math_tokens or has_calc_word)


# ---------------------------------------------------------------------------
# Coordinator class
# ---------------------------------------------------------------------------

from rich.console import Console

from agents._llm import make_agent, run_with_fallback
from agents._prompts import ROUTER_SYSTEM
from rag.embedder import Embedder
from schemas.routing import RoutingDecision

console = Console()


class Coordinator:
    """
    Routes queries to the right specialist agent.

    Flow:
      1. Arithmetic pre-filter -> general agent with calculator (no LLM routing call).
      2. LLM router -> RoutingDecision -> dispatch to retriever or general.
      3. 'unclear' queries get a clarification response (not a guess).
      4. 'mixed' queries run retriever then general; results are bundled.
    """

    def __init__(self, conn, embedder: Embedder) -> None:
        from agents.general import GeneralAgent
        from agents.retriever import RetrieverAgent
        self._retriever = RetrieverAgent(conn=conn, embedder=embedder)
        self._general = GeneralAgent()

    def route(self, query: str) -> RoutingDecision:
        if looks_like_arithmetic(query):
            return RoutingDecision(
                query_type="calculation",
                reasoning=(
                    "Arithmetic pre-filter matched: query contains numbers and "
                    "math operators with no reasoning verbs. Routed directly to "
                    "the general agent so it can invoke the calculator tool."
                ),
                selected_agent="general_agent",
                tools_likely_needed=["calculator"],
                requires_followup=False,
                followup_description=None,
            )

        def builder(primary: bool):
            return make_agent(
                system=ROUTER_SYSTEM,
                response_model=RoutingDecision,
                primary=primary,
            )

        prompt = (
            f"Classify this query and return a RoutingDecision:\n\n"
            f"{query}\n\n"
            f"Remember: reasoning must be at least 20 characters and explain "
            f"why other types were ruled out."
        )

        try:
            raw = run_with_fallback(builder, prompt)
            decision = raw.content if hasattr(raw, "content") else raw
            if isinstance(decision, RoutingDecision):
                return decision
            # Some models return plain text instead of structured output — default to doc_qa
            raise ValueError(f"Expected RoutingDecision, got {type(decision)}")
        except Exception as exc:
            return RoutingDecision(
                query_type="document_qa",
                reasoning=(
                    f"Routing LLM call failed ({exc.__class__.__name__}). Defaulting "
                    f"to document_qa as the safe fallback — preferable to guessing "
                    f"between general_knowledge and web_search without an LLM."
                ),
                selected_agent="retriever_agent",
                tools_likely_needed=[],
                requires_followup=False,
                followup_description=None,
            )

    def handle(self, query: str) -> dict:
        """End-to-end: route, run agent(s), return a bundle the CLI can render."""
        decision = self.route(query)

        if decision.query_type == "unclear":
            return {
                "decision": decision,
                "agent": "none",
                "result": None,
                "clarification": (
                    "I'm not sure what you're asking. Could you rephrase? "
                    "I can search the document knowledge base, do math, "
                    "or answer general questions."
                ),
            }

        if decision.query_type == "mixed":
            retrieval = self._retriever.answer(query)
            chunk_summary = "\n".join(
                f"- {c.text[:200]}" for c in retrieval.retrieved_chunks[:3]
            )
            general = self._general.answer(
                f"Given these document excerpts:\n{chunk_summary}\n\nAnswer: {query}"
            )
            return {
                "decision": decision,
                "agent": "mixed",
                "retrieval": retrieval,
                "general": general,
            }

        if decision.selected_agent == "retriever_agent":
            result = self._retriever.answer(query)
        else:
            result = self._general.answer(query)

        return {
            "decision": decision,
            "agent": decision.selected_agent,
            "result": result,
        }


# ---------------------------------------------------------------------------
# Smoke test (requires API key and ingested DB)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()
    from rag.store import get_store

    DB = "data/research_assistant.duckdb"
    if not Path(DB).exists():
        console.print(f"[red]Missing {DB}. Run `python ingest.py` first.[/red]")
        raise SystemExit(1)

    embedder = Embedder()
    conn = get_store(DB)
    coord = Coordinator(conn=conn, embedder=embedder)

    # Pre-filter check (no API needed)
    console.rule("[bold cyan]Pre-filter check[/bold cyan]")
    decision = coord.route("12 * 8 + sqrt(81)")
    console.print(f"Pre-filter -> {decision.query_type} (expected: calculation)")
