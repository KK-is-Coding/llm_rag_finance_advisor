"""
shared/llm.py
=============

Centralizing `HuggingFaceHub(...)` object here means:
    - the HF token is validated once, with a clear error if missing
    - if you swap models later, you only change it in one place

Used by:
    - economic_report/rag_pipeline.py
    - news_sentiment/pipeline.py
    - investment_advisor/rag_pipeline.py

NOTE: `HuggingFaceHub` (from `langchain_community.llms`) is
deprecated. The current replacement is `HuggingFaceEndpoint` from the
dedicated `langchain_huggingface` package. Make sure
`langchain-huggingface` is installed.

NOTE 2: As of mid-2025, HF Inference Providers serve most modern
instruct/chat models (Qwen, Llama, etc.) only under the
"conversational" task, not the older "text-generation" (raw
completion) task. Calling `HuggingFaceEndpoint` directly uses the
completion API and fails with:
    ValueError: Model ... is not supported for task text-generation
    and provider ...  Supported task: conversational.
The fix is to wrap the endpoint in `ChatHuggingFace`, which calls the
chat-completion API instead. This returns `AIMessage` objects (not
plain strings) from `.invoke()` — callers use `.content` to get text.
"""

import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from config import (
    HUGGINGFACEHUB_API_TOKEN,
    LLM_REPO_ID,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    LLM_TIMEOUT_SECONDS,
)

_llm_instance = None


def get_llm() -> ChatHuggingFace:
    """Returning a lazily-created, shared ChatHuggingFace LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        if not HUGGINGFACEHUB_API_TOKEN:
            raise EnvironmentError(
                "HUGGINGFACEHUB_API_TOKEN is not set. Add it to your .env file "
                "or export it in your shell before running this pipeline."
            )

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

        endpoint = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            task="conversational",
        )
        _llm_instance = ChatHuggingFace(llm=endpoint)
    return _llm_instance