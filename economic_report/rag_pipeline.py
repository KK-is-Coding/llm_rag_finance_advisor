import os
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import ECONOMIC_CSV_PATH
from shared.embeddings import get_embeddings
from shared.llm import get_llm
from shared.vectorstore import split_documents, build_chroma_store

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
    return split_documents(documents, chunk_size=200, chunk_overlap=20)


def build_vectorstore():
    """
    Embed the split documents into an in-memory (non-persisted) Chroma store.

    This pipeline re-fetches and rebuilds from scratch for a different
    ticker on every call — there's nothing worth persisting to disk,
    and doing so previously left behind a growing pile of stale
    per-run Chroma folders under docs/chroma_economic/ for no benefit.
    """
    texts = load_and_split_csv()
    embeddings = get_embeddings()

    return build_chroma_store(
        documents=texts,
        embeddings=embeddings,
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