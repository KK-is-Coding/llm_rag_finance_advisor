"""
news_sentiment/pipeline.py
============================
Step 2: turning the fetched news headlines into a single prompt
and ask the LLM to analyze sentiment/impact.

IMPORTANT — The notebook never embeds the news articles or does a similarity
search on them; it just concatenates every headline into one big
prompt and sends it straight to the LLM. That's fine for a handful of
articles but will blow past the model's context window if `page_size`
in fetch_news.py is increased. Flagging this here so it's not confused
with the other two (real) RAG pipelines when debugging.

Depends on:
    - news_sentiment/fetch_news.py
    - shared/llm.py

"""

from urllib import response

from click import prompt
import pandas as pd

from shared import llm
from shared.llm import get_llm
from news_sentiment.fetch_news import fetch_and_preprocess

PROMPT_HEADER = (
    "You are a financial analyst tasked with providing insights into recent "
    "news articles related to the financial industry. Here are some recent "
    "news articles:\n\n"
)
PROMPT_FOOTER = (
    "Please analyze these articles and provide insights into any potential "
    "impacts on the financial industry sentiment on the provided company."
)


def build_prompt(news_df: pd.DataFrame) -> str:
    """Concatenate headlines into a single analysis prompt."""
    lines = [f"   **News:** {title}\n" for title in news_df["title"]]
    return PROMPT_HEADER + "\n".join(lines) + "\n" + PROMPT_FOOTER


def run_sentiment_summary(query: str, days_back: int = 10) -> str:
    """Full step-2 pipeline: fetch news -> build prompt -> call LLM."""
    news_df = fetch_and_preprocess(query, days_back=days_back)
    if news_df.empty:
        return "No news articles found — nothing to summarize."

    prompt = build_prompt(news_df)
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    summary = run_sentiment_summary("Microsoft News")
    # in case only if this file is run directly, not if imported as a module
    print(summary)
