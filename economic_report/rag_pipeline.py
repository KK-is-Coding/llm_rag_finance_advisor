"""
economic_report/rag_pipeline.py
=================================
Project 1, Step 2: load the CSV produced by fetch_data.py, embed it,
store it in Chroma, and run a RetrievalQA chain against it to produce
a financial report answer.

This corresponds to notebook cells 11, 13, 14, 15, 17, 18.

Depends on:
    - economic_report/fetch_data.py  (must run first, produces the CSV)
    - shared/embeddings.py
    - shared/llm.py
    - shared/vectorstore.py

Run directly with:  python -m economic_report.rag_pipeline
"""

import os
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import ECONOMIC_CSV_PATH, CHROMA_DIR_ECONOMIC
from shared.embeddings import get_embeddings
from shared.llm import get_llm
from shared.vectorstore import split_documents, build_chroma_store, clear_collection

REPORT_TEMPLATE = """You are a Financial Market Expert. Use the Market Economic
Data below to build a financial report for the requested company.
Context: {context}
Question: {question}
Please present the key figures in a table."""


def load_and_split_csv(csv_path: str = ECONOMIC_CSV_PATH):
    """Load the economic-indicator CSV and split it into chunks."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m economic_report.fetch_data` first."
        )
    loader = CSVLoader(csv_path)
    documents = loader.load()
    # NOTE: chunk_size=50 (as in the original notebook) is extremely small for
    # this data — most CSV rows will get split mid-field. Bumped to a saner
    # default; override if you know your row sizes.
    return split_documents(documents, chunk_size=200, chunk_overlap=20)


def build_vectorstore():
    """Embed the split documents and persist them to Chroma."""
    texts = load_and_split_csv()
    embeddings = get_embeddings()

    # IMPORTANT: this pipeline is re-run for a different ticker every time
    # (fetch_data.py overwrites eco_ind.csv each call). Without clearing the
    # collection first, Chroma silently appends the new ticker's data to
    # whatever was already persisted from previous tickers/runs, and the
    # retriever can then mix up data between companies. Always clear first.
    clear_collection(CHROMA_DIR_ECONOMIC, embeddings, collection_name="economic_data")

    return build_chroma_store(
        documents=texts,
        embeddings=embeddings,
        persist_directory=CHROMA_DIR_ECONOMIC,
        collection_name="economic_data",
    )


def run_report_query(question: str, vectordb=None, k: int = 2) -> str:
    """Run the RAG chain for a given natural-language question."""
    vectordb = vectordb or build_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=REPORT_TEMPLATE
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        retriever=retriever,
        return_source_documents=True,
    )
    response = qa_chain({"query": question})
    return response["result"]


if __name__ == "__main__":
    answer = run_report_query("Microsoft(MSFT) Financial Report")
    print(answer)