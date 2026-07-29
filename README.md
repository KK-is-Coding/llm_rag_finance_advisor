# 💰 AI Financial Advisor — RAG-Powered Investment Insights 🤖

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Inference-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> 📈 Ask it about your money. It'll actually go check, instead of guessing.

An AI-powered financial advisory tool that combines three data sources —
peer investor behavior 👥, live market data 📊, and current news sentiment 📰 —
into one synthesized investment recommendation, using Retrieval-Augmented
Generation (RAG) and a large language model.

Instead of asking an LLM to guess investment advice from its training data
alone, this project grounds every answer in real, retrievable data: actual
market quotes fetched live, actual recent news headlines, and actual
patterns from a dataset of past investor decisions — then asks the LLM to
reason over that evidence rather than invent it. ✅ Grounded, not guessed.

---

## ✨ What it does

You describe yourself (age, gender, preferred investment type, goal, time
horizon) and name up to 3 stock tickers you're considering. The tool then:

1. 👥 **Finds similar past investors** — searches a database of investor
   profiles for people with similar goals, and surfaces what worked for them
2. 📊 **Pulls live financial data** — current price, volume, market cap,
   moving averages, and more for each ticker you named
3. 📰 **Checks recent news sentiment** — recent headlines about each company
4. 🧠 **Synthesizes one recommendation** — combines all three into a single,
   readable answer: a side-by-side comparison table (if multiple tickers),
   an individual verdict for each one, and an overall recommendation
   tailored to your stated goals

---

## 🗂️ Project structure

```
llm_rag_finance/
├── config.py                    # ⚙️  all settings/API keys, loaded from env vars
├── .env.example                 # 🔑 template — copy to .env and fill in your keys
├── .gitignore
├── requirements.txt
├── main.py                      # 🚀 entry point — run this to start the advisor
├── data/                        # 📁 generated/downloaded data files live here
│   └── .gitkeep
│
├── shared/                      # 🧩 code shared across all pipelines
│   ├── embeddings.py            #     the embedding model (singleton)
│   ├── llm.py                   #     the language model (singleton)
│   └── vectorstore.py           #     split + embed + store helpers (Chroma)
│
├── economic_report/              # 📊 live financial data pipeline
│   ├── fetch_data.py            #     pulls a stock quote from Financial Modeling Prep
│   └── rag_pipeline.py          #     turns that data into a readable report
│
├── news_sentiment/                # 📰 news pipeline
│   ├── fetch_news.py            #     pulls recent headlines from NewsAPI
│   └── pipeline.py               #     summarizes sentiment via the LLM
│
├── investment_advisor/           # 👥 peer investor advice pipeline
│   ├── data_prep.py             #     loads the investor dataset, builds
│   │                             #     searchable documents from it
│   ├── generate_synthetic_data.py  # 🧪 optional: generate fake data for testing
│   └── rag_pipeline.py          #     retrieves similar profiles, generates advice
│
└── final_advisor/                # 🧠 combines all three into one app
    └── pipeline.py               #     the actual advisor you interact with
```

---

## ✅ Prerequisites

- 🐍 Python 3.11 or later
- Free accounts (all have free tiers) for:
  - 🔑 [Financial Modeling Prep](https://site.financialmodelingprep.com/register) — market data
  - 🤗 [Hugging Face](https://huggingface.co/join) — the language model
  - 📰 [NewsAPI](https://newsapi.org/register) — news headlines
- 📊 A dataset of investor profiles for the peer-advice pipeline — either:
  - the real dataset from [Kaggle](https://www.kaggle.com/datasets/nitindatta/finance-data), or
  - a synthetic one generated for you (see setup below) — good for testing the
    pipeline immediately, but not real investor behavior

---

## 🛠️ Setup

**1️⃣ Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**2️⃣ Set up your API keys:**
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `FMP_API_KEY` — from your Financial Modeling Prep dashboard
- `HUGGINGFACEHUB_API_TOKEN` — from Hugging Face → Settings → Access Tokens (Read access is enough)
- `NEWSAPI_API_KEY` — from your NewsAPI account page

> ⚠️ Never commit your real `.env` file — it's already excluded in `.gitignore`.

**3️⃣ Get the investor dataset — pick one:**

🅰️ *Use the real dataset (recommended for genuine results):*
Download the CSV from [Kaggle](https://www.kaggle.com/datasets/nitindatta/finance-data)
(from the dataset's **Data** tab — not the "download metadata" option) and
save it as `data/Finance_data.csv`.

🅱️ *Generate a synthetic dataset to test the pipeline immediately:*
```bash
python -m investment_advisor.generate_synthetic_data
```
This creates a plausible fake dataset at `data/Finance_data.csv` so you can
run the whole app right away. Swap in the real dataset later for genuine
investment insights — synthetic data is for testing the pipeline mechanics
only, not real advice. 🧪

---

## 🚀 Running the project

```bash
python main.py
```

This starts an interactive session: it asks about you, asks which
ticker(s) you want researched, then runs the full pipeline and prints a
combined recommendation.

### 🖥️ Example session
*(illustrative — your real output will vary with live data)*

```
Let's build your investor profile.

Your age: 28
Gender (Male/Female): Female
Preferred investment avenue (e.g. Mutual Fund, Equity, Fixed Deposits, Government Bonds): Mutual Fund, Equity
Purpose of investing (e.g. Wealth Creation, Retirement Plan, Rainy Days, Educational Requirements): Wealth Creation
Investment duration (e.g. Less than 1 year, 1-3 years, 3-5 years, More than 5 years): 3-5 years

Which stock ticker(s) do you want researched/compared? (comma-separated, up to 3, e.g. MSFT,AAPL,GOOGL): MSFT,AAPL

[1/3] Getting peer-based reference advice...
[2/3] (1/2) Researching MSFT...
[2/3] (2/2) Researching AAPL...
[3/3] Synthesizing final recommendation...

=== FINAL RECOMMENDATION ===

### Investment Recommendation for the Investor

#### Investor's Stated Profile
- Age: 28
- Gender: Female
- Preferred investment avenue(s): Mutual Fund, Equity
- Purpose: Wealth Creation
- Investment duration: 3-5 years

#### Company Comparison

| Metric              | MSFT       | AAPL       |
|----------------------|-----------|-----------|
| Current Price         | $415.20   | $221.85   |
| Change                | +1.2%     | -0.4%     |
| Market Capitalization | $3.08T    | $3.41T    |
| 50-Day Moving Average | $402.10   | $228.60   |

#### Individual Verdicts

- **MSFT**: Trading above its moving average with steady momentum;
  suitable for a growth-oriented allocation given the investor's
  3-5 year horizon.
- **AAPL**: Slight recent pullback but strong long-term fundamentals;
  a reasonable stability anchor within an equity allocation.

#### On Preferred Avenues (Mutual Fund, Equity)

Given the wealth-creation goal and moderate-length horizon, a mix of
direct equity (such as the tickers above) and a diversified mutual
fund is a reasonable balance — equity for growth potential, mutual
funds to diversify away single-stock risk.

### Overall Recommendation

**Best Fit: A combination of MSFT and AAPL, complemented by a
diversified mutual fund allocation.**

This balances the investor's stated preference for both direct equity
and mutual funds, while managing single-stock risk over a 3-5 year
horizon.
```

*(Numbers above are illustrative placeholders — real runs pull live prices,
sentiment, and peer data at the moment you run them.)* 📉📈

---

## 🔧 Running individual pieces (for testing/debugging)

Each stage can also be run on its own:

```bash
python -m economic_report.fetch_data       # 📊 pull + save market data for one ticker
python -m economic_report.rag_pipeline     # 📄 generate a report from that data
python -m news_sentiment.pipeline          # 📰 summarize news sentiment for a company
python -m investment_advisor.rag_pipeline  # 👥 get peer-based advice for a hardcoded example question
```

---

## 🧠 How it's built

- 🔍 **RAG (Retrieval-Augmented Generation)**: investor profiles and market
  data are embedded into a vector database (Chroma) and retrieved by
  similarity to your question, so the LLM answers using real retrieved
  evidence rather than only its training knowledge.
- 🤗 **Language model**: served via Hugging Face's Inference Providers — no
  local GPU or model download required.
- 🌐 **Live data**: market quotes come from Financial Modeling Prep; news from
  NewsAPI — both fetched fresh on each run.

---

## ⚠️ Notes and limitations

- 🌍 Free-tier market data covers US-listed tickers and ADRs; some
  internationally-listed stocks (e.g. NSE/BSE-only listings) require a paid
  data plan and won't resolve on the free tier.
- 🚫 This tool is for learning/prototyping a RAG pipeline — it is **not**
  financial advice, and outputs should not be used for real investment
  decisions without independent verification.
- 📊 The peer-advice pipeline is only as good as the dataset behind it —
  synthetic data is useful for testing the mechanics, but only the real
  dataset reflects genuine investor behavior.

---

<p align="center">Made with ☕ and a healthy amount of debugging.</p>