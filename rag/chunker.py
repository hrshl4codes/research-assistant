"""
Text chunker — splits raw document text into overlapping fragments.

Uses LangChain's RecursiveCharacterTextSplitter (chunk_size=1000,
chunk_overlap=150) so chunks respect sentence boundaries rather than
splitting mid-word.  Overlap ensures a sentence spanning two chunks
appears in at least one of them.

Not yet implemented.
"""
