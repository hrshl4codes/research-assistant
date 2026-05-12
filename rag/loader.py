"""
Document loader — extracts text from PDFs and plain-text files.

Uses PyMuPDF (fitz) for PDFs.  Returns a list of raw page strings
alongside metadata (source filename, page number) that propagates into
chunk records so retrieved context can be traced back to the original doc.

Not yet implemented.
"""
