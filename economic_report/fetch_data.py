"""
economic_report/fetch_data.py
==============================
Project 1, Step 1: pull a stock quote / economic indicators from the
Financial Modeling Prep (FMP) API, clean it up, and save it to CSV.

This corresponds to notebook cells 2, 5, 7.

Bugs fixed vs. the notebook:
1. The original function signature was
       def get_jsonparsed_data(url, api_key, exchange)
   but it never used `url` — it silently overwrote it using a *global*
   `ticker` variable defined outside the function. That's fragile (the
   function only works if a same-named global happens to exist in
   whatever scope calls it). Here `ticker` is a proper parameter.
2. The notebook hit `/api/v3/...` — FMP retired those "legacy"
   endpoints for anyone without a subscription that predates
   Aug 31 2025; that's what causes the `HTTP Error 403: Forbidden`.
   Current endpoints live under `/stable/`. Updated below.
3. `urlopen(url, cafile=...)` is deprecated in Python 3.12+ in favor
   of passing an `ssl.SSLContext`. Switched to `context=`.

Run directly with:  python -m economic_report.fetch_data
Output:  data/eco_ind.csv  (consumed by economic_report/rag_pipeline.py)
"""

import os
import ssl
import json
import certifi
import pandas as pd
from urllib.request import urlopen
from urllib.error import HTTPError

from config import FMP_API_KEY, ECONOMIC_CSV_PATH, DATA_DIR

FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Reuse one SSL context instead of re-resolving certifi's cert bundle on
# every call (and avoids the `cafile=` deprecation warning).
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def get_jsonparsed_data(ticker: str, api_key: str, exchange: str) -> dict:
    """Call the FMP API for a given ticker and return parsed JSON."""
    if exchange == "NSE":
        url = f"{FMP_BASE_URL}/search-symbol?query={ticker}&exchange=NSE&apikey={api_key}"
    else:
        url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={api_key}"

    try:
        response = urlopen(url, context=_SSL_CONTEXT)
    except HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                "FMP returned 403 Forbidden. Most likely causes: "
                "(1) your API key's plan doesn't include this endpoint, "
                "or (2) the key is invalid/expired. Log into "
                "https://site.financialmodelingprep.com/developer/docs/dashboard "
                "and confirm the key + plan, then retry."
            ) from e
        if e.code == 402:
            raise RuntimeError(
                f"FMP returned 402 Payment Required for ticker '{ticker}' "
                f"(exchange={exchange}). This usually means the ticker is on a "
                f"non-US exchange (e.g. NSE/BSE-listed Indian stocks like TCS) "
                f"and that data is gated behind a paid FMP plan. Try a "
                f"US-listed ticker or ADR instead (e.g. INFY has a US ADR, "
                f"TCS does not), or upgrade your FMP plan for international data."
            ) from e
        raise

    data = response.read().decode("utf-8")
    return json.loads(data)


def preprocess_economic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp-like columns to real datetimes.

    The notebook assumed both `timestamp` and `earningsAnnouncement`
    would always be present, which was true of the old `/api/v3/quote`
    response. The current `/stable/quote` response only returns
    `timestamp` — `earningsAnnouncement` now lives on FMP's separate
    earnings-calendar endpoint, not the quote endpoint. Rather than
    re-hardcode a column list that FMP can change again, this only
    converts whichever of these columns actually showed up.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    if "earningsAnnouncement" in df.columns:
        df["earningsAnnouncement"] = pd.to_datetime(
            df["earningsAnnouncement"], errors="coerce"
        )
    return df


def fetch_and_save(ticker: str = "MSFT", exchange: str = "US") -> pd.DataFrame:
    """Full step-1 pipeline: fetch -> preprocess -> save CSV -> return df."""
    if not FMP_API_KEY:
        raise EnvironmentError("FMP_API_KEY is not set. See .env.example.")

    raw = get_jsonparsed_data(ticker, FMP_API_KEY, exchange)
    eco_ind = pd.DataFrame(raw)

    preprocessed = preprocess_economic_data(eco_ind)

    os.makedirs(DATA_DIR, exist_ok=True)
    preprocessed.to_csv(ECONOMIC_CSV_PATH, index=False)
    print(f"Saved {len(preprocessed)} rows to {ECONOMIC_CSV_PATH}")

    return preprocessed


if __name__ == "__main__":
    df = fetch_and_save(ticker="MSFT", exchange="US")
    print(df.head())