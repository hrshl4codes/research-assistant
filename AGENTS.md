# Research Assistant — Agent System

Three agents handle all queries. The Coordinator decides which agent runs; no agent calls another directly. The Retriever answers from documents. The General agent answers from knowledge and tools.

---

## Coordinator

**File:** `agents/coordinator.py`  
**Schema:** `schemas/routing.py` — `RoutingDecision`

### Arithmetic pre-filter

Before the LLM is called, a regex check runs on every query. If it matches, the query goes straight to the General agent as `calculation` with no LLM routing call.

A query matches if:
- It contains at least one number
- It contains a math token (`+`, `-`, `*`, `/`, `**`, `sqrt`, `sin`, `cos`, `tan`, `log`, `ln`, `exp`, `pi`, `e`) or a trigger word (`compute`, `calculate`, `what is`)
- It is 80 characters or shorter
- It does not contain an explanatory verb (`explain`, `describe`, `compare`, `summarize`, `why`, `how does`, `tell me about`, `who`)

False negatives fall through to the LLM router. False positives are caught by the calculator's own error handling.

### LLM routing

When the pre-filter does not match, the Coordinator calls the LLM with a `RoutingDecision` response model. The six query types are:

| Type | Meaning |
|---|---|
| `document_qa` | Needs a lookup in the document knowledge base |
| `general_knowledge` | Open-domain factual question, no documents needed |
| `calculation` | Needs the calculator tool |
| `web_search` | Needs current or external information |
| `mixed` | Needs both retrieval and a tool or general step |
| `unclear` | Intent cannot be determined |

`RoutingDecision.reasoning` has a minimum length of 20 characters enforced by Pydantic. A routing decision with no stated reason fails validation immediately.

### Failure behavior

- If the LLM returns malformed output or raises, the Coordinator defaults to `document_qa` (conservative, not a guess).
- If `query_type` is `unclear`, the Coordinator returns a clarification message and does not call any agent.

### Mixed query flow

1. RetrieverAgent runs first.
2. The top 3 retrieved chunks (truncated to 200 chars each) are passed as context to the GeneralAgent.
3. Both results are returned in a bundle for the CLI to render.

---

## RetrieverAgent

**File:** `agents/retriever.py`  
**Schema:** `schemas/results.py` — `RetrievalResult`

### Retrieval

- The system embeds the query with `all-MiniLM-L6-v2` (384 dimensions).
- It searches the HNSW index in DuckDB for top-k=4 chunks by cosine distance.
- Chunks are formatted as a numbered block with source and distance annotations, then passed to the LLM.

### Answer rules

The LLM is instructed to:
1. Answer using ONLY the provided chunks. No general knowledge fallback.
2. Cite chunk numbers like `[chunk 1]`.
3. If chunks do not address the question, say so plainly.

### Confidence

Confidence comes from the top chunk's cosine distance, not from the model's self-report:

| Distance | Confidence |
|---|---|
| Below 0.30 | `high` |
| 0.30 to 0.50 | `medium` |
| Above 0.50 | `low` |

### Failure behavior

- No chunks returned by the vector store: returns `RetrievalResult` with `confidence=low` and `refusal_reason` set. `retrieved_chunks` is empty.
- LLM call fails: returns a refusal result with the exception message in `refusal_reason`.
- LLM returns an empty response: returns a refusal result rather than an empty answer (which would fail schema validation).
- `retrieved_chunks` is always populated when chunks exist, even on refusal.

---

## GeneralAgent

**File:** `agents/general.py`  
**Schema:** `schemas/results.py` — `GeneralResult`

### Tools

`calculator_tool`
- Input: `expression` (string), `context` (optional string)
- Passes the expression to sympy for symbolic evaluation.
- Returns the simplified form and numeric value when finite.
- The LLM is instructed to always use this tool for math. Never compute in its head.

`web_search_tool`
- Input: `query_str` (string), `max_results` (int, 1-10)
- Returns results from a static mock dataset covering RAG, transformers, and federated learning.
- Returns an empty result for queries outside that set.
- Results are labeled `source="mock_data"`.

### Tool audit

Tool calls are recorded by closure wrappers, not by asking the model what it used. `GeneralResult.tools_used` comes from `call_log`, which is appended to only when a tool actually executes. The LLM's narration of what it did is not used.

Each `ToolCall` record contains: `tool_name`, `input`, `output`, `duration_ms`, `success`.

### Failure behavior

- LLM call fails: returns `GeneralResult` with an empty answer and the exception in `reasoning_trace`.
- Tools that error return an `ERROR: ...` string to the LLM rather than raising.

---

## Shared behavior

### Tracing

Every `Coordinator.route`, `Coordinator.handle`, `RetrieverAgent.answer`, and `GeneralAgent.answer` call appends a line to `logs/trace.jsonl` via the `@trace` decorator. Each line records the component name, inputs, outputs, and wall-clock duration.

### LLM fallback

Every agent call goes through `run_with_fallback`. Primary model is `gpt-4o-mini`. If it raises, the call retries with the fallback model (`gpt-3.5-turbo`). Both are configurable via `PRIMARY_MODEL` and `FALLBACK_MODEL` in `.env`.

### Response cache

`_tracing.py` provides `cache_get` and `cache_put`. Cache keys are SHA-256 hashes of `(component, query)`. Hits skip the LLM call entirely.

### Singletons

`get_embedder()` and `get_store_conn()` return process-wide singletons. The sentence-transformer model and DuckDB connection each load once per process.
