"""
Web search tool — mocked, but transparent in output.

Returns a structured SearchResponse that clearly labels results as mocked
and includes the query that would have been issued.  The mock surfaces the
tool's intended I/O contract so it can be swapped for a real search API
(SerpAPI, Brave, etc.) without touching the agent code.

Not yet implemented.  Input/output types defined in schemas/tools.py.
"""
