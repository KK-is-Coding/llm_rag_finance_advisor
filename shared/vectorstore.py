"""
shared/vectorstore.py
======================
Generic helpers for building and querying a Chroma vector store.

Both RAG pipelines (economic_report and investment_advisor) did the
same three steps in the notebook: split docs -> embed -> store in
Chroma. This module extracts that repeated pattern into one function
so a bug fix here fixes it everywhere, instead of having to patch two
near-identical code blocks separately.

Used by:
    - economic_report/rag_pipeline.py
    - investment_advisor/rag_pipeline.py

NOTE: LangChain split into several packages in 2024
(langchain-core / langchain-community / langchain-text-splitters /
langchain-chroma etc). The notebook was written against the old
single-package layout, so these imports point at the current homes
of `RecursiveCharacterTextSplitter` and `Chroma` rather than
`langchain.text_splitter` / `langchain.vectorstores`, which no
longer exist. Make sure `requirements.txt` is installed (it lists
the split-out packages) or these imports will fail.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def split_documents(documents, chunk_size=100, chunk_overlap=10):
    """Split loaded documents into smaller chunks before embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def clear_collection(persist_directory, embeddings, collection_name=None):
    """
    Delete an existing Chroma collection at this path/name, if any.

    `Chroma.from_documents()` APPENDS to an existing persisted
    collection rather than replacing it. For data that changes between
    calls (e.g. economic_report re-fetching a different ticker every
    time), that silently accumulates old rows from every previous
    ticker/run into the same collection, and the retriever can then
    return context from the WRONG source. Call this before rebuilding
    to guarantee the collection only reflects the current data.
    """
    try:
        Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name,
        ).delete_collection()
    except Exception:
        pass  # nothing existed to delete yet — fine on first run


def build_chroma_store(
    documents, embeddings, persist_directory, collection_name=None
):
    """
    Embed `documents` and store them in a Chroma collection on disk.

    Returns the Chroma vector store object, ready for
    `.similarity_search()` or `.as_retriever()`.
    """
    kwargs = dict(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    if collection_name:
        kwargs["collection_name"] = collection_name

    return Chroma.from_documents(**kwargs)