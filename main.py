"""
main.py
=======
Single entry point that runs all three pipelines in order. Useful for
smoke-testing the whole project and for seeing exactly how the files
connect to each other end-to-end.

Data flow
---------
Project 1 — Economic Report:
    economic_report/fetch_data.py   --(writes CSV)-->  data/eco_ind.csv
    economic_report/rag_pipeline.py --(reads CSV, uses shared/embeddings,
                                        shared/llm, shared/vectorstore)-->
                                        Chroma DB --> RetrievalQA answer

Project 2 — News Sentiment:
    news_sentiment/fetch_news.py  --(returns DataFrame)-->
    news_sentiment/pipeline.py    --(uses shared/llm, NOT a RAG chain,
                                      just a direct prompt)--> LLM summary

Project 3 — Investment Advisor:
    investment_advisor/data_prep.py   --(reads data/Finance_data.csv,
                                          returns Documents)-->
    investment_advisor/rag_pipeline.py --(uses shared/embeddings,
                                           shared/llm, shared/vectorstore)-->
                                           Chroma DB --> RetrievalQA answer

Run individual pieces instead of everything with, e.g.:
    python -m economic_report.fetch_data
    python -m economic_report.rag_pipeline
    python -m news_sentiment.pipeline
    python -m investment_advisor.rag_pipeline
"""

from economic_report.fetch_data import fetch_and_save
from economic_report.rag_pipeline import run_report_query
from news_sentiment.pipeline import run_sentiment_summary
from investment_advisor.rag_pipeline import run_advisor_query


def run_economic_report():
    print("\n=== Project 1: Economic Report RAG ===")
    fetch_and_save(ticker="MSFT", exchange="US")
    answer = run_report_query("Microsoft(MSFT) Financial Report")
    print(answer)


def run_news_sentiment():
    print("\n=== Project 2: News Sentiment Summarizer ===")
    summary = run_sentiment_summary("Microsoft News")
    print(summary)


def run_investment_advisor():
    print("\n=== Project 3: Investment Advisor RAG ===")
    result = run_advisor_query(
        "I'm a 34-year-old female looking to invest in mutual funds for "
        "wealth creation over the next 1-3 years. What are my options?"
    )
    print(result["result"])


if __name__ == "__main__":
    run_economic_report()
    run_news_sentiment()
    run_investment_advisor()
