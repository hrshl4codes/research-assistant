"""
Smoke test for Agno + OpenRouter integration.

Run with:
    python main.py

Expected output: a single short response from the LLM proving the API
key, the Agno model adapter, and the OpenRouter endpoint all work.
"""

print("This is a smoke test for the LLM connection. For the actual assistant, run python assistant.py.")

import os
import sys

from dotenv import load_dotenv
from rich.console import Console

console = Console()


def _load_env() -> str:
    """Load .env and return the OpenRouter API key, raising if absent."""
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    return key


def run_smoke_test() -> None:
    """Make a single test call to OpenRouter via Agno and print the response."""
    _load_env()

    # Import after env is loaded so Agno picks up the key from the environment.
    from agno.agent import Agent
    from agno.models.openrouter import OpenRouter

    primary_model = os.getenv(
        "PRIMARY_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    fallback_model = os.getenv(
        "FALLBACK_MODEL", "google/gemini-2.0-flash-exp:free"
    )

    console.rule("[bold cyan]Agno + OpenRouter Smoke Test[/bold cyan]")
    console.print(f"[dim]Primary model :[/dim] {primary_model}")
    console.print(f"[dim]Fallback model :[/dim] {fallback_model}")
    console.print()

    agent = Agent(
        model=OpenRouter(id=primary_model),
        # Keep system prompt minimal so a free-tier model doesn't time out.
        description="You are a terse assistant.",
        markdown=False,
    )

    # A deterministic prompt makes it easy to confirm the response is real.
    probe = (
        "Reply with exactly this sentence and nothing else: "
        "'Agno + OpenRouter smoke test passed.'"
    )

    console.print("[bold]Sending probe to primary model…[/bold]")
    try:
        agent.print_response(probe, stream=False)
        console.print("\n[bold green]Smoke test PASSED.[/bold green]")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"\n[yellow]Primary model failed ({exc}). "
            f"Trying fallback: {fallback_model}[/yellow]"
        )
        fallback_agent = Agent(
            model=OpenRouter(id=fallback_model),
            description="You are a terse assistant.",
            markdown=False,
        )
        fallback_agent.print_response(probe, stream=False)
        console.print("\n[bold green]Smoke test PASSED (via fallback).[/bold green]")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except EnvironmentError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        sys.exit(1)
