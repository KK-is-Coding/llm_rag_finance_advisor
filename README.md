# LLM + RAG for Finance — restructured from the notebook

This is the original `LLM + RAG for Finance.ipynb` notebook (by
Simranjeet Singh) split into a proper Python project, one concern per
file, so each piece can be run and debugged independently instead of
re-running a giant notebook top to bottom.

## Folder structure

```
llm_rag_finance/
├── config.py                    # all settings/API keys, loaded from env vars
├── .env.example                 # template for your local .env file
├── requirements.txt
├── main.py                      # runs all 3 pipelines end-to-end
├── data/                        # eco_ind.csv gets written here; put
│                                 #   Finance_data.csv here manually
├── shared/                      # code reused by more than one pipeline
│   ├── embeddings.py            #   HuggingFaceEmbeddings (singleton)
│   ├── llm.py                   #   Falcon-7B-Instruct via HF Hub (singleton)
│   └── vectorstore.py           #   generic "split + embed + store in Chroma"
├── economic_report/             # Project 1 (notebook cells 1-18)
│   ├── fetch_data.py            #   FMP API -> preprocess -> CSV
│   └── rag_pipeline.py          #   CSV -> Chroma -> RetrievalQA
├── news_sentiment/               # Project 2 (notebook cells 19-26)
│   ├── fetch_news.py            #   NewsAPI -> preprocess
│   └── pipeline.py               #   headlines -> prompt -> direct LLM call
├── investment_advisor/          # Project 3 (notebook cells 27-38)
│   ├── data_prep.py             #   Kaggle CSV -> prompt/response Documents
│   └── rag_pipeline.py          #   Documents -> Chroma -> RetrievalQA
└── fraud_detection/              # Project 4 heading only, no code in notebook
    └── README.md
```

## How the pieces connect

Each project is a two-step pipeline: **get/prepare data → run the LLM
against it.** The `shared/` folder holds the parts identical across
projects (the embedding model and the LLM) so there's exactly one
place to fix things like a bad API key or a model swap.

```
fetch_data.py / fetch_news.py / data_prep.py   (data in)
              │
              ▼
      shared/vectorstore.py  +  shared/embeddings.py   (Project 1 & 3 only)
              │
              ▼
        shared/llm.py  →  RetrievalQA / direct prompt
              │
              ▼
            answer
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in your own keys
```

## Running

```bash
# run one project at a time
python -m economic_report.fetch_data
python -m economic_report.rag_pipeline
python -m news_sentiment.pipeline
python -m investment_advisor.rag_pipeline

# or run everything
python main.py
```

Note: `investment_advisor` needs `data/Finance_data.csv` downloaded
manually from
[Kaggle](https://www.kaggle.com/datasets/nitindatta/finance-data) —
the notebook never fetched it via API either.

---

## ⚠️ Important: rotate your API keys

The notebook you uploaded had **three real API keys hardcoded directly
in the code cells**:

- an FMP (Financial Modeling Prep) key
- a HuggingFace Hub token
- a NewsAPI key

Anyone who received or viewed that `.ipynb` file could read and use
those keys. I did not carry them into this project — every key is now
read from environment variables via `config.py` and `.env`. **You
should treat all three of the original keys as compromised and
regenerate them** on each service's dashboard, then put the new ones
in your `.env` file.

## Other bugs fixed while restructuring

1. **`get_jsonparsed_data(url, api_key, exchange)`** — the `url`
   parameter was never used; the function silently rebuilt the URL
   using a *global* `ticker` variable from outside the function. Now
   takes `ticker` as an explicit parameter (`economic_report/fetch_data.py`).
2. **Duplicate model loading** — `HuggingFaceEmbeddings()` and
   `HuggingFaceHub(...)` were each instantiated twice with identical
   arguments in different cells. Centralized as singletons in
   `shared/embeddings.py` and `shared/llm.py`.
3. **Shared Chroma folder for two different datasets** — the notebook
   reused `docs/chroma_rag/`-style paths loosely; Project 1 and
   Project 3 now get separate persist directories
   (`CHROMA_DIR_ECONOMIC` / `CHROMA_DIR_ADVISOR`) so their embeddings
   never mix.
4. **`df.drop("source", axis=1)` could raise `KeyError`** if NewsAPI
   ever returns a response without that column — now guarded with a
   column-existence check (`news_sentiment/fetch_news.py`).
5. **Very small `chunk_size`** (50 for economic data, 100 for advisor
   data) caused CSV rows/answers to be split mid-field, hurting
   retrieval quality. Increased to more reasonable defaults — tune
   further based on your actual row sizes.
6. **Missing dependency check** — `investment_advisor/data_prep.py`
   now validates that `Finance_data.csv` has all the columns the code
   expects and raises a clear error instead of a raw `KeyError`
   partway through a loop.
7. **`HTTP Error 403: Forbidden` from FMP** — the notebook's
   `/api/v3/...` endpoints were retired by Financial Modeling Prep for
   any account without a subscription predating Aug 31 2025.
   `economic_report/fetch_data.py` now calls the current `/stable/`
   endpoints instead. If you still get a 403 after this fix, it means
   your specific API key/plan doesn't include that endpoint — check
   your plan on the FMP dashboard.
8. **`KeyError: 'earningsAnnouncement'`** — a side effect of #7. The
   new `/stable/quote` endpoint returns a different field set than the
   old `/api/v3/quote` did, and no longer includes
   `earningsAnnouncement` (that data now lives on FMP's separate
   earnings-calendar endpoint). `preprocess_economic_data()` now only
   converts columns that are actually present instead of assuming a
   fixed schema.

## Note on the News Sentiment project

Unlike Project 1 and Project 3, **Project 2 is not actually RAG.** The
notebook never embeds the news articles into a vector store or does a
similarity search on them — it just joins every fetched headline into
one long prompt and sends it straight to the LLM. That's preserved
as-is in `news_sentiment/pipeline.py`, with a note in the docstring,
since fixing that would change the pipeline's behavior rather than
just its structure. If you want it to scale past a handful of
articles, that's the piece to convert into a real retriever.
