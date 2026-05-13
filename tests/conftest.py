# tests/conftest.py
"""Shared test fixtures."""
import pytest

from rag.store import SearchResult


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Three plausible retrieval results with descending relevance."""
    return [
        SearchResult(
            id="abc123",
            doc_id="sample.pdf",
            text="Federated learning trains models across devices without "
                 "centralizing data, preserving privacy.",
            source="sample.pdf",
            cosine_distance=0.12,
        ),
        SearchResult(
            id="def456",
            doc_id="sample.pdf",
            text="Differential privacy adds Gaussian noise to gradients before "
                 "they leave each device.",
            source="sample.pdf",
            cosine_distance=0.34,
        ),
        SearchResult(
            id="ghi789",
            doc_id="sample.pdf",
            text="Communication efficiency is a key bottleneck in federated systems.",
            source="sample.pdf",
            cosine_distance=0.48,
        ),
    ]
