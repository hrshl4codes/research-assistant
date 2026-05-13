"""
Central LLM client factory. All agents go through this so model selection,
fallback behavior, and base configuration are defined in one place.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

_log = logging.getLogger(__name__)

from agno.agent import Agent
from agno.models.openrouter import OpenRouter


def _model_id(primary: bool = True) -> str:
    if primary:
        return os.getenv("PRIMARY_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    return os.getenv("FALLBACK_MODEL", "google/gemini-2.0-flash-exp:free")


def make_agent(
    *,
    system: str,
    response_model: type | None = None,
    tools: list[Callable] | None = None,
    primary: bool = True,
) -> Agent:
    """
    Build an Agno Agent with our project conventions.

    Args:
        system: System/description prompt the agent runs under.
        response_model: Pydantic class for structured output. None = free-form text.
        tools: List of Python callables registered as Agno tools.
        primary: Use the primary model id (False uses the fallback).

    Returns:
        Configured Agno Agent ready for .run() or .print_response().
    """
    return Agent(
        model=OpenRouter(id=_model_id(primary)),
        description=system,
        response_model=response_model,
        tools=tools or [],
        markdown=False,
    )


def run_with_fallback(builder: Callable[[bool], Agent], prompt: str) -> Any:
    """
    Try the primary model; on any exception, retry with the fallback.

    `builder` is a function taking `primary: bool` and returning a configured Agent.
    """
    try:
        agent = builder(True)
        return agent.run(prompt)
    except Exception as primary_exc:
        _log.warning("Primary model failed (%s: %s), retrying with fallback", type(primary_exc).__name__, primary_exc)
        agent = builder(False)
        return agent.run(prompt)
