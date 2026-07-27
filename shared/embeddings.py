"""
shared/embeddings.py
=====================
One place that creates the HuggingFace embedding model.

In the original notebook, `HuggingFaceEmbeddings()` was instantiated
twice (once as `hg_embeddings`, once again as `embeddings` a few cells
later) — both are the exact same model, loaded twice for no reason.
This module makes sure it's only ever loaded once and re-used
everywhere (economic_report, investment_advisor, etc.).

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
    """Return a lazily-created, shared HuggingFaceEmbeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings()
    return _embeddings_instance