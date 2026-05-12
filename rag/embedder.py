"""
Embedder — wraps sentence-transformers for local, free embedding generation.

Model: all-MiniLM-L6-v2 (384 dimensions).
Downloads on first use; subsequent calls load from the local model cache.
Returns normalised float32 vectors (required for cosine distance to be
meaningful without separate normalisation).

Not yet implemented.
"""
