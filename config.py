"""
config.py
=========
Single source of truth for all settings and API keys.

Every other module imports from here instead of hardcoding credentials.
This is the #1 fix vs. the original notebook, which had three live API
keys typed directly into code cells (FMP key, HuggingFace Hub token,
NewsAPI key). That means anyone who opened the .ipynb file could see
and use them.

Set these as real environment variables (or put them in a local
`.env` file — see `.env.example`) before running anything else in
this project.
"""

import os

# ---- Optional: auto-load a local .env file if python-dotenv is installed ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can also be set directly in the shell


def _get_required(name: str) -> str:
    """Fetch an env var and fail loudly (not silently) if it's missing."""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Set it in your shell or in a .env file (see .env.example)."
        )
    return value


# --- Financial Modeling Prep (economic_report pipeline) ---
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

# --- HuggingFace Hub (used by every pipeline that calls an LLM) ---
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

# --- NewsAPI (news_sentiment pipeline) ---
NEWSAPI_API_KEY = os.environ.get("NEWSAPI_API_KEY", "")

# --- Shared model config ---
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "Qwen/Qwen2.5-72B-Instruct")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
# How many tokens the model is allowed to generate per response. The default
# on HF Inference Providers can be quite short (a few hundred tokens), which
# cuts off longer outputs like multi-ticker comparison tables mid-sentence.
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "2048"))
# How long (seconds) to wait for the model to respond before giving up. Larger
# models (like the 72B one) generating longer responses (max_new_tokens above)
# can genuinely take a while, especially on a "cold" provider connection —
# the previous default was too short and caused "read operation timed out"
# errors that looked like failures but were really just impatience.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "180"))

# --- Vector DB persistence location for investment_advisor ---
# (economic_report intentionally does NOT persist — see
# economic_report/rag_pipeline.py for why)
CHROMA_DIR_ADVISOR = os.environ.get("CHROMA_DIR_ADVISOR", "docs/chroma_advisor/")

# --- Data file locations ---
DATA_DIR = os.environ.get("DATA_DIR", "data/")
ECONOMIC_CSV_PATH = os.path.join(DATA_DIR, "eco_ind.csv")
FINANCE_ADVISOR_CSV_PATH = os.environ.get(
    "FINANCE_ADVISOR_CSV_PATH", os.path.join(DATA_DIR, "Finance_data.csv")
)