"""
Tool I/O schemas — Pydantic models for calculator and web_search inputs/outputs.

Structured I/O ensures:
- Agents produce parseable, validatable tool calls (not free-form text).
- Tool outputs carry provenance (expression, query) alongside the result
  so the agent can cite them in its final response.

Not yet implemented.
"""
