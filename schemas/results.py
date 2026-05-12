"""
Agent output schemas. Typed responses returned by RetrieverAgent and GeneralAgent.

RetrievalResult carries the answer plus the retrieved evidence so the caller
can show it to the user. Surfacing retrieved chunks is a hard requirement —
a system that consumes retrieved evidence silently provides no transparency
about where its answer came from.

GeneralResult carries the answer plus a full tool audit trail and a
reasoning trace so every step the agent took is inspectable.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from schemas.tools import ToolCall

# Confidence levels and their intended thresholds (set by the agent):
#   high   — top retrieved chunk cosine_distance < 0.30
#   medium — top retrieved chunk cosine_distance 0.30–0.50
#   low    — top retrieved chunk cosine_distance > 0.50, or fewer chunks
#            than expected, or answer required significant inference
Confidence = Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """
    A single piece of evidence retrieved from the vector store.

    This is a presentation-layer model — it carries only what the user
    and agent need to see, not internal store fields like the chunk id.
    """

    text: str = Field(..., description="The raw chunk text, shown verbatim to the user.")
    source: str = Field(
        ...,
        description="Human-readable citation, e.g. 'sample.pdf, page 2'.",
    )
    cosine_distance: float = Field(
        ...,
        ge=0.0,
        le=2.0,
        description=(
            "Distance from the query vector. 0 = identical; 2 = opposite. "
            "Lower means more relevant."
        ),
    )
    doc_id: str = Field(
        ...,
        description="The source document filename. Useful for grouping chunks by document.",
    )


# ---------------------------------------------------------------------------
# RetrieverAgent output
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """
    Output of the RetrieverAgent. Answer grounded in retrieved document evidence.

    retrieved_chunks must be populated even if the answer is a refusal —
    it shows the user what was (or wasn't) found, which is itself informative.
    """

    query: str = Field(..., description="The original user query, echoed for traceability.")
    answer: str = Field(
        ...,
        description=(
            "The agent's answer. If the retrieved evidence does not support "
            "a confident answer, this should say so explicitly rather than "
            "hallucinating — set refusal_reason in that case."
        ),
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        ...,
        description=(
            "The chunks retrieved from the vector store, in ranked order "
            "(most similar first). Must not be empty — if nothing was retrieved, "
            "the retriever should explain that in the answer and set confidence=low."
        ),
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "Agent's assessment of answer quality. Set based on cosine distances "
            "and whether the retrieved chunks directly address the query: "
            "high (<0.30 top distance), medium (0.30–0.50), low (>0.50)."
        ),
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description=(
            "If the agent could not produce a grounded answer, explain why here. "
            "None when the agent answered normally."
        ),
    )

    @model_validator(mode="after")
    def refusal_needs_reason(self) -> "RetrievalResult":
        """An empty answer without a refusal_reason is a silent failure."""
        if not self.answer.strip() and not self.refusal_reason:
            raise ValueError(
                "answer is empty but refusal_reason is not set. "
                "Either provide an answer or explain why one cannot be given."
            )
        return self


# ---------------------------------------------------------------------------
# GeneralAgent output
# ---------------------------------------------------------------------------


class GeneralResult(BaseModel):
    """
    Output of the GeneralAgent. Answer produced via reasoning and optional tool use.

    tools_used provides a complete audit trail: what was called, with what
    arguments, what came back, and how long each call took. An agent that
    used no tools returns an empty list, not a null field.
    """

    query: str = Field(..., description="The original user query, echoed for traceability.")
    answer: str = Field(..., description="The agent's final answer to the query.")
    tools_used: list[ToolCall] = Field(
        default_factory=list,
        description=(
            "Ordered list of tool calls made during this response. "
            "Empty when the agent answered from parametric knowledge alone."
        ),
    )
    reasoning_trace: str = Field(
        ...,
        description=(
            "Step-by-step account of how the agent reached the answer. "
            "Should name what the agent knew, what it looked up, and how "
            "it combined them. This is what separates genuine reasoning "
            "from prompt-chained output."
        ),
    )
