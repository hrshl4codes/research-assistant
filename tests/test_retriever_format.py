# tests/test_retriever_format.py
"""Unit tests for RetrieverAgent's deterministic helpers."""
import pytest

from agents.retriever import (
    format_chunks_for_prompt,
    confidence_from_distance,
    search_results_to_retrieved_chunks,
)


def test_format_chunks_for_prompt(sample_search_results):
    formatted = format_chunks_for_prompt(sample_search_results)
    assert "[chunk 1]" in formatted
    assert "[chunk 2]" in formatted
    assert "[chunk 3]" in formatted
    assert "sample.pdf" in formatted
    assert "0.12" in formatted or "0.120" in formatted


def test_confidence_high_for_close_distance():
    assert confidence_from_distance(0.10) == "high"
    assert confidence_from_distance(0.29) == "high"


def test_confidence_medium_for_mid_distance():
    assert confidence_from_distance(0.30) == "medium"
    assert confidence_from_distance(0.49) == "medium"


def test_confidence_low_for_far_distance():
    assert confidence_from_distance(0.50) == "low"
    assert confidence_from_distance(1.50) == "low"


def test_confidence_low_for_no_chunks():
    assert confidence_from_distance(None) == "low"


def test_search_results_to_retrieved_chunks(sample_search_results):
    chunks = search_results_to_retrieved_chunks(sample_search_results)
    assert len(chunks) == 3
    assert chunks[0].cosine_distance == 0.12
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].doc_id == "sample.pdf"
    assert "Federated learning" in chunks[0].text
