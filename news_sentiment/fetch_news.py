"""
news_sentiment/fetch_news.py
==============================
Step 1: pulling recent news articles from NewsAPI about a
company/topic and clean the result down to (author, title) pairs.

"""

import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta

from config import NEWSAPI_API_KEY


def fetch_news(
    query: str,
    from_date,
    to_date,
    language: str = "en",
    sort_by: str = "relevancy",
    page_size: int = 30,
    api_key: str = None,
) -> pd.DataFrame:
    """Fetching articles from NewsAPI matching `query` in the given date range."""
    api_key = api_key or NEWSAPI_API_KEY
    if not api_key:
        raise EnvironmentError("NEWSAPI_API_KEY is not set. See .env.example.")

    newsapi = NewsApiClient(api_key=api_key)
    query = query.replace(" ", "&")

    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language=language,
        sort_by=sort_by,
        page_size=page_size,
    )

    articles = all_articles.get("articles", [])
    return pd.DataFrame(articles) if articles else pd.DataFrame()


def preprocess_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without an author, keep just author + title."""
    df = df.copy()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df = df[~df["author"].isna()]
    return df[["author", "title"]]


def fetch_and_preprocess(query: str, days_back: int = 10) -> pd.DataFrame:
    """Full step-1 pipeline: fetch -> drop 'source' col -> preprocess."""
    current_time = datetime.now()
    start_time = current_time - timedelta(days=days_back)

    raw_df = fetch_news(query, start_time, current_time)
    if raw_df.empty:
        print("No articles found for this query/date range.")
        return raw_df

    if "source" in raw_df.columns:
        raw_df = raw_df.drop("source", axis=1)

    return preprocess_news_data(raw_df)


if __name__ == "__main__":
    df = fetch_and_preprocess("Microsoft News") # in case if onlyif this file is run directly, not if imported as a module
    print(df.head())
