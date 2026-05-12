"""
Calculator tool. Safe symbolic math via sympy.

Parses expressions using sympy's own parser (never Python's built-in
code execution) and returns a symbolic simplified form plus a numeric
approximation. Handles parse errors, undefined results, and division
by zero without crashing.
"""

from __future__ import annotations

from typing import Optional

import sympy
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CalculatorInput(BaseModel):
    """What the agent sends to the calculator tool."""

    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate, e.g. '2**10 + sqrt(16)'.",
    )
    context: str = Field(
        default="",
        description=(
            "Why this calculation is being performed. "
            "The agent includes this so the tool result can be cited clearly."
        ),
    )


class CalculatorOutput(BaseModel):
    """What the calculator tool returns to the agent."""

    expression: str = Field(..., description="Original expression, echoed back.")
    context: str = Field(..., description="Original context, echoed back.")
    result: str = Field(
        ...,
        description="Sympy's evaluated form as a string, e.g. '1024 + 2*sqrt(2)'.",
    )
    simplified: str = Field(
        ...,
        description=(
            "Fully simplified form. Differs from result for trig identities, "
            "algebraic simplifications, etc."
        ),
    )
    numeric_value: Optional[float] = Field(
        default=None,
        description=(
            "Float approximation. None when the expression has free variables, "
            "evaluates to infinity, or is otherwise non-numeric."
        ),
    )
    success: bool
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message. None when success is True.",
    )


# ---------------------------------------------------------------------------
# Whitelisted names for the sympy parser
# ---------------------------------------------------------------------------
# parse_expr resolves names against this dict. Anything not listed here
# (e.g. '__import__', 'open') raises a NameError at parse time, not at
# runtime, so the restriction is enforced before any computation starts.

_ALLOWED_NAMES: dict = {
    "sqrt": sympy.sqrt,
    "log": sympy.log,
    "ln": sympy.log,
    "log2": lambda x: sympy.log(x, 2),
    "log10": lambda x: sympy.log(x, 10),
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "factorial": sympy.factorial,
    "ceiling": sympy.ceiling,
    "floor": sympy.floor,
    "pi": sympy.pi,
    "e": sympy.E,
    "oo": sympy.oo,
}

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


def calculate(inp: CalculatorInput) -> CalculatorOutput:
    """
    Parse and evaluate a mathematical expression using sympy.

    Args:
        inp: CalculatorInput with the expression and optional context.

    Returns:
        CalculatorOutput. On parse or math error, success=False and
        error contains the message; result and simplified are empty strings.
    """

    def _fail(msg: str) -> CalculatorOutput:
        return CalculatorOutput(
            expression=inp.expression,
            context=inp.context,
            result="",
            simplified="",
            numeric_value=None,
            success=False,
            error=msg,
        )

    expr_str = inp.expression.strip()
    if not expr_str:
        return _fail("Expression is empty.")

    # Parse — restricted to whitelisted names only.
    try:
        parsed = parse_expr(
            expr_str,
            local_dict=_ALLOWED_NAMES,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        return _fail(f"Could not parse '{expr_str}': {exc}")

    # Simplify — catches things like sin²+cos²→1.
    try:
        simplified = sympy.simplify(parsed)
    except Exception as exc:
        return _fail(f"Simplification failed: {exc}")

    # zoo = complex infinity (e.g. 1/0 in sympy). Treat as an error rather
    # than returning a confusing "zoo" string to the agent.
    if simplified is sympy.zoo or simplified == sympy.zoo:
        return _fail("Expression is undefined (division by zero or complex infinity).")

    # Numeric approximation — only when no free symbols remain and the
    # result is finite.
    numeric_value: Optional[float] = None
    if not simplified.free_symbols and simplified not in (sympy.oo, -sympy.oo, sympy.nan):
        try:
            numeric_value = float(sympy.N(simplified, 15))
        except Exception:
            pass  # complex result or otherwise non-float — leave as None

    return CalculatorOutput(
        expression=inp.expression,
        context=inp.context,
        result=str(parsed),
        simplified=str(simplified),
        numeric_value=numeric_value,
        success=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold cyan]Calculator Tool — Self-Test[/bold cyan]")

    tests = [
        CalculatorInput(
            expression="2**10 + sqrt(16)",
            context="Sanity check: power-of-two plus square root",
        ),
        CalculatorInput(
            expression="(1 + sqrt(5)) / 2",
            context="Golden ratio for a layout scaling formula",
        ),
        CalculatorInput(
            expression="sin(pi/6)**2 + cos(pi/6)**2",
            context="Verifying the Pythagorean trig identity equals 1",
        ),
        CalculatorInput(
            expression="1 / 0",
            context="Testing division-by-zero error path",
        ),
    ]

    table = Table(show_lines=True, expand=True)
    table.add_column("Expression", style="bold", width=26)
    table.add_column("Simplified", width=18)
    table.add_column("Numeric value", width=16, justify="right")
    table.add_column("OK?", width=5, justify="center")
    table.add_column("Error / Context", width=36, style="dim")

    for t in tests:
        out = calculate(t)
        table.add_row(
            t.expression,
            out.simplified if out.success else "",
            f"{out.numeric_value:.8f}" if out.numeric_value is not None else "—",
            "[green]✓[/green]" if out.success else "[red]✗[/red]",
            out.error if not out.success else t.context[:40],
        )

    console.print(table)
