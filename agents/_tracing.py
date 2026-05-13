"""
Lightweight JSONL tracing. Every agent .answer()/.route() call appends a single
JSON line to logs/trace.jsonl with timing, inputs, and outputs.

Also provides process-wide singletons for the embedder and store connection,
plus a disk-backed response cache keyed by SHA-256 of (component, query).
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "trace.jsonl"
_CACHE_DIR = Path("logs/cache")


def _safe_dump(value: Any) -> Any:
    """Best-effort JSON-serializable representation."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_safe_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_dump(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def trace(component: str) -> Callable:
    """Decorator: log a JSONL line with timing and serialized inputs/outputs."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            err: str | None = None
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                err = repr(exc)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                LOG_DIR.mkdir(exist_ok=True)
                entry = {
                    "ts": time.time(),
                    "component": component,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "args": _safe_dump(args[1:]) if args else [],
                    "kwargs": _safe_dump(kwargs),
                    "result": _safe_dump(result),
                    "error": err,
                }
                with LOG_FILE.open("a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Process-wide singletons
# ---------------------------------------------------------------------------

_embedder = None
_conn = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from rag.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def get_store_conn(db_path: str = "data/research_assistant.duckdb"):
    global _conn
    if _conn is None:
        from rag.store import get_store
        _conn = get_store(db_path)
    return _conn


# ---------------------------------------------------------------------------
# Disk-backed response cache
# ---------------------------------------------------------------------------

def cache_key(component: str, query: str) -> str:
    h = hashlib.sha256(f"{component}::{query}".encode()).hexdigest()[:16]
    return h


def cache_get(component: str, query: str) -> dict | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = _CACHE_DIR / f"{cache_key(component, query)}.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def cache_put(component: str, query: str, value: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = _CACHE_DIR / f"{cache_key(component, query)}.json"
    f.write_text(json.dumps(value, default=str, indent=2))
