"""
Shared tool invocation record.

ToolCall is not a call spec — it is a stamped record of a call that already
happened. The agent creates one after each tool invocation so the result
carries a full audit trail: what was called, with what input, what came back,
and how long it took.

input/output are dict[str, Any] rather than the tool's own Pydantic types.
This keeps schemas/ independent of tools/ and avoids circular imports. Callers
serialize tool inputs/outputs with .model_dump() before storing them here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal["calculator", "web_search"]


class ToolCall(BaseModel):
    """A record of one tool invocation, attached to a GeneralResult."""

    tool_name: ToolName = Field(
        ...,
        description="Which tool was called.",
    )
    input: dict[str, Any] = Field(
        ...,
        description=(
            "Serialized tool input (the tool's Input Pydantic model as a dict). "
            "Preserves the exact arguments the agent passed."
        ),
    )
    output: dict[str, Any] = Field(
        ...,
        description=(
            "Serialized tool output (the tool's Output Pydantic model as a dict). "
            "Includes success/error state from the tool itself."
        ),
    )
    duration_ms: float = Field(
        ...,
        ge=0,
        description="Wall-clock time for the tool call in milliseconds.",
    )
    success: bool = Field(
        ...,
        description=(
            "True if the tool completed without error. "
            "Mirrors the success field in the tool's own output schema."
        ),
    )
