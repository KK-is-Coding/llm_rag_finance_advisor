"""
shared/embeddings.py
=====================

Used by:
    - economic_report/rag_pipeline.py
    - investment_advisor/rag_pipeline.py

NOTE: `HuggingFaceEmbeddings` used to live in `langchain_community`;
it's now provided by the dedicated `langchain_huggingface` package
(langchain-community still re-exports it but prints a deprecation
warning). Make sure `langchain-huggingface` is installed.
"""

from langchain_huggingface import HuggingFaceEmbeddings

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Returning a lazily-created, shared HuggingFaceEmbeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings()
    return _embeddings_instance