# Agent Skills

What each agent can do, what inputs it accepts, and what it returns.

---

## Coordinator

**Entry point:** `Coordinator.handle(query: str) -> dict`

### Arithmetic pre-filter

Detects pure math queries without an LLM call. Matches queries that have a number, a math token or trigger word, are under 80 characters, and contain no explanatory verbs. Bypasses LLM routing when matched.

### Query classification

Classifies any query that passes the pre-filter into one of six types: `document_qa`, `general_knowledge`, `calculation`, `web_search`, `mixed`, or `unclear`. Returns a `RoutingDecision` with explicit reasoning.

### Agent dispatch

Routes `document_qa` and `mixed` to the RetrieverAgent. Routes `general_knowledge`, `calculation`, and `web_search` to the GeneralAgent. Returns a result bundle the CLI can render directly.

---

## RetrieverAgent

**Entry point:** `RetrieverAgent.answer(query: str) -> RetrievalResult`

### Query embedding

Encodes the query into a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers). Runs locally with no API call.

### Vector search

Searches the HNSW index in DuckDB for the top 4 chunks by cosine distance. Returns `SearchResult` objects with text, source, distance, and document ID.

### Grounded answer generation

The agent formats retrieved chunks as a numbered block and sends them to the LLM with a strict instruction to answer only from those chunks. Chunk numbers are cited in the response.

### Confidence scoring

The agent maps the top chunk's cosine distance to a confidence label (`high`, `medium`, `low`). It does not ask the model how confident it is.

### Refusal

When chunks are empty, when the LLM fails, or when the model returns an empty response, returns a `RetrievalResult` with `confidence=low` and `refusal_reason` set. Always includes whatever chunks were retrieved, even in failure cases.

---

## GeneralAgent

**Entry point:** `GeneralAgent.answer(query: str) -> GeneralResult`

### Parametric knowledge

Answers open-domain questions from the model's training data when no tool is needed.

### Calculator

Calls `calculator_tool(expression, context)` for arithmetic and symbolic math. Uses sympy under the hood. Returns the simplified expression and numeric value when finite. Handles errors by returning an `ERROR: ...` string rather than raising.

### Web search

Calls `web_search_tool(query_str, max_results)` for current-event or external lookups. Returns results from a static mock dataset. Covered topics: RAG, transformers, federated learning. Returns empty for anything outside that set.

### Tool audit

The agent wraps each tool in a closure that records to `call_log` on every actual execution. `GeneralResult.tools_used` is built from that log. If the model narrates using a tool it did not call, the audit log catches the discrepancy.

### Reasoning trace

The agent populates `GeneralResult.reasoning_trace` with a step-by-step account of what it knew, what it looked up, and how it combined them.

---

## Shared skills

### Tracing

All four entry points (`Coordinator.route`, `Coordinator.handle`, `RetrieverAgent.answer`, `GeneralAgent.answer`) are decorated with `@trace`. Each call writes a JSONL line to `logs/trace.jsonl` with the component name, inputs, outputs, and duration.

### Response caching

Any traced call can be cached by key `(component, query)` using SHA-256 hashing. Cache hits return stored results without calling the LLM.

### LLM fallback

All LLM calls go through `run_with_fallback(builder, prompt)`. The builder is called with `primary=True` first, then `primary=False` on failure. Models are set via `PRIMARY_MODEL` and `FALLBACK_MODEL` in `.env`.
