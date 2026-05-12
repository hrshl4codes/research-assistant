"""
General agent — answers queries that don't require document retrieval.

Responsibilities:
- Handle open-domain questions using the LLM's parametric knowledge.
- Invoke the calculator tool for arithmetic/symbolic math.
- Invoke the web_search tool for current-events lookups.

Not yet implemented.  Depends on tools/calculator.py and tools/web_search.py.
"""
