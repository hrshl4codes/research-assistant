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
