"""
investment_advisor/data_prep.py
=================================
Project 3, Step 1: load the Kaggle "Finance_data.csv" dataset and turn
each row into a (prompt, response) pair describing an investor profile
and the advice given to them.

Data source referenced in the notebook:
https://www.kaggle.com/datasets/nitindatta/finance-data
You must download this CSV yourself and place it at the path in
config.FINANCE_ADVISOR_CSV_PATH (data/Finance_data.csv by default) —
it isn't fetched automatically by any API.

This corresponds to notebook cells 29, 30, 32.

Run directly with:  python -m investment_advisor.data_prep
"""

import os
import pandas as pd
from langchain_core.documents import Document
from config import FINANCE_ADVISOR_CSV_PATH


REQUIRED_COLUMNS = [
    "age", "gender", "Avenue", "Purpose", "Duration",
    "Mutual_Funds", "Equity_Market", "Debentures", "Government_Bonds",
    "Fixed_Deposits", "PPF", "Gold", "Factor", "Objective", "Expect",
    "Invest_Monitor", "Reason_Equity", "Reason_Mutual", "Reason_Bonds",
    "Reason_FD", "Source",
]


def load_finance_data(csv_path: str = FINANCE_ADVISOR_CSV_PATH) -> list[dict]:
    """Load the raw CSV into a list of row dicts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Download it from "
            f"https://www.kaggle.com/datasets/nitindatta/finance-data "
            f"and place it at that path."
        )

    # encoding="utf-8-sig" strips a leading BOM if present (common in CSVs
    # re-saved on Windows/Excel) — a BOM otherwise gets glued onto the first
    # column name and can throw off parsing.
    # sep=None + engine="python" auto-detects the delimiter instead of
    # assuming comma, in case the file was re-saved with ';' or tabs.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")

    # Strip stray whitespace from column names (e.g. " age" vs "age").
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Finance_data.csv is missing expected columns: {missing}\n"
            f"Columns actually found in the file: {list(df.columns)}\n"
            f"Compare these two lists to see if it's a naming mismatch "
            f"(e.g. different capitalization or spelling) vs. a genuinely "
            f"different dataset version."
        )

    return df.to_dict(orient="records")


def build_prompt_response_pairs(data_fin: list[dict]) -> list[dict]:
    """Convert each investor row into a {prompt, response} training-style pair."""
    pairs = []
    for entry in data_fin:
        prompt = (
            f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest "
            f"in {entry['Avenue']} for {entry['Purpose']} over the next "
            f"{entry['Duration']}. What are my options?"
        )
        response = (
            f"Based on your preferences, here are your investment options:\n"
            f"- Mutual Funds: {entry['Mutual_Funds']}\n"
            f"- Equity Market: {entry['Equity_Market']}\n"
            f"- Debentures: {entry['Debentures']}\n"
            f"- Government Bonds: {entry['Government_Bonds']}\n"
            f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
            f"- PPF: {entry['PPF']}\n"
            f"- Gold: {entry['Gold']}\n"
            f"Factors considered: {entry['Factor']}\n"
            f"Objective: {entry['Objective']}\n"
            f"Expected returns: {entry['Expect']}\n"
            f"Investment monitoring: {entry['Invest_Monitor']}\n"
            f"Reasons for choices:\n"
            f"- Equity: {entry['Reason_Equity']}\n"
            f"- Mutual Funds: {entry['Reason_Mutual']}\n"
            f"- Bonds: {entry['Reason_Bonds']}\n"
            f"- Fixed Deposits: {entry['Reason_FD']}\n"
            f"Source of information: {entry['Source']}\n"
        )
        pairs.append({"prompt": prompt, "response": response})
    return pairs


def build_documents(pairs: list[dict]) -> list[Document]:
    """Turn prompt-response pairs into LangChain Document objects for embedding."""
    documents = []
    for entry in pairs:
        combined_text = f"Prompt: {entry['prompt']}\nResponse: {entry['response']}"
        documents.append(Document(page_content=combined_text))
    return documents


if __name__ == "__main__":
    data_fin = load_finance_data()
    pairs = build_prompt_response_pairs(data_fin)
    print(f"Built {len(pairs)} prompt-response pairs. Example:\n")
    print(pairs[0]["prompt"])
    print(pairs[0]["response"])
