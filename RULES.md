# Agent Rules

Hard constraints that govern every agent in the system. These are enforced by code, not by asking the model to comply.

---

## Coordinator rules

1. Always run the arithmetic pre-filter before calling the LLM. No exceptions.
2. Never route without a stated reason. `RoutingDecision.reasoning` has a 20-character minimum enforced by Pydantic; a bare classification without explanation fails validation.
3. On LLM failure, default to `document_qa`. Do not guess between `general_knowledge` and `web_search`.
4. On `unclear`, return a clarification message. Do not forward an unclear query to any agent.
5. On `mixed`, run the RetrieverAgent first. Pass chunk summaries to the GeneralAgent. Do not run them in parallel.
6. Never call RetrieverAgent or GeneralAgent directly from outside the Coordinator. All dispatch goes through `Coordinator.handle`.

---

## RetrieverAgent rules

1. Answer using only the chunks provided in the prompt. No fallback to general knowledge.
2. Always cite which chunk the answer came from, using `[chunk N]` notation.
3. If chunks do not address the question, say so. Do not invent an answer.
4. Never return an empty answer without a `refusal_reason`. The schema validator will reject it.
5. Always populate `retrieved_chunks` in the response, even when refusing. An empty chunk list with no explanation is not acceptable.
6. Set confidence from cosine distance, not from the model's own assessment:
   - Below 0.30: `high`
   - 0.30 to 0.50: `medium`
   - Above 0.50: `low`

---

## GeneralAgent rules

1. Always use `calculator_tool` for math. Never compute arithmetic in the model's head.
2. Use `web_search_tool` for questions about current events or external lookups.
3. `tools_used` is populated from the closure audit log, not from the model's narration of what it did.
4. On tool error, the tool returns an `ERROR: ...` string to the model. The model must not raise.
5. Populate `reasoning_trace` with a step-by-step account. A one-sentence trace is not acceptable.

---

## System-wide rules

1. Every agent call is traced. `@trace` appends a line to `logs/trace.jsonl` on every invocation.
2. All agent inputs and outputs are typed. `RoutingDecision`, `RetrievalResult`, and `GeneralResult` are Pydantic models. Untyped responses are not accepted.
3. LLM calls go through `run_with_fallback`. The primary model (`gpt-4o-mini`) is tried first. On failure, the fallback (`gpt-3.5-turbo`) is tried. When both fail, the calling agent returns a typed failure result, not a bare exception.
4. The embedder and DuckDB connection are process-wide singletons. Do not instantiate them more than once.
5. Retrieved chunks are always shown to the user. The system must not consume retrieved evidence silently.
