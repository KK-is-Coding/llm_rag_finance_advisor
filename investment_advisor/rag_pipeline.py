"""
investment_advisor/rag_pipeline.py
====================================
Project 3, Step 2: embed the prompt-response documents from
data_prep.py, store them in Chroma, and run a RetrievalQA chain so the
LLM answers new investor questions using similar past examples as
context.

This corresponds to notebook cells 34, 35, 36, 38.

Depends on:
    - investment_advisor/data_prep.py
    - shared/embeddings.py
    - shared/llm.py
    - shared/vectorstore.py

Run directly with:  python -m investment_advisor.rag_pipeline
"""

from langchain_classic.chains import RetrievalQA

from config import CHROMA_DIR_ADVISOR
from shared.embeddings import get_embeddings
from shared.llm import get_llm
from shared.vectorstore import split_documents, build_chroma_store, clear_collection
from investment_advisor.data_prep import (
    load_finance_data,
    build_prompt_response_pairs,
    build_documents,
)


def build_vectorstore():
    """Full ingestion pipeline: load CSV -> build docs -> split -> embed -> store."""
    data_fin = load_finance_data()
    pairs = build_prompt_response_pairs(data_fin)
    documents = build_documents(pairs)

    texts = split_documents(documents, chunk_size=500, chunk_overlap=50)
    # NOTE: original notebook used chunk_size=100 here, which is small enough
    # to cut most of these multi-line responses into fragments that don't
    # retrieve well together. Bumped up; tune to your data.

    embeddings = get_embeddings()

    # Clear any previously persisted collection first — otherwise re-running
    # this (e.g. from combined_advisor calling it repeatedly, or just running
    # the script again another day) keeps appending the same rows on top of
    # what's already there, duplicating documents in the collection.
    clear_collection(CHROMA_DIR_ADVISOR, embeddings)

    return build_chroma_store(
        documents=texts,
        embeddings=embeddings,
        persist_directory=CHROMA_DIR_ADVISOR,
    )



def run_advisor_query(question: str, vectordb=None, k: int = 5) -> dict:
    """Run the RAG chain for a given investor question."""
    vectordb = vectordb or build_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain({"query": question})


if __name__ == "__main__":
    result = run_advisor_query(
        "I'm a 34-year-old female looking to invest in mutual funds for "
        "wealth creation over the next 1-3 years. What are my options?"
    )
    print(result["result"])
