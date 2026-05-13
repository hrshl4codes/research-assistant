# tests/test_coordinator_prefilter.py
"""
The coordinator should short-circuit obvious arithmetic queries to the general
agent (with calculator) without burning an LLM call on routing.
"""

import pytest

from agents.coordinator import looks_like_arithmetic


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what is 2+2", True),
        ("compute 2**10 + sqrt(16)", True),
        ("(1 + sqrt(5)) / 2", True),
        ("12.5 * 8", True),
        ("calculate sin(pi/4)", True),
        ("what is the document about", False),
        ("who wrote the paper", False),
        ("explain RAG", False),
        ("the number 42 is the answer to what question", False),
        ("compare federated learning to centralized training", False),
    ],
)
def test_arithmetic_prefilter(query, expected):
    assert looks_like_arithmetic(query) is expected
