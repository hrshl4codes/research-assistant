"""
Retriever agent — answers queries that require grounded document context.

Responsibilities:
- Embed the user query and retrieve top-k chunks from the DuckDB VSS store.
- Expose the retrieved chunks explicitly in the response (not silently consumed).
- Generate an answer that cites which chunk(s) it drew from.

Not yet implemented.  Depends on rag/store.py and rag/embedder.py.
"""
