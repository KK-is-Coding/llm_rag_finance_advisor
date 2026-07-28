"""
main.py
=======
Single entry point for the whole project.

This used to duplicate final_advisor/pipeline.py's logic by calling
each of the three sub-pipelines separately and printing three
disconnected answers. That duplication is exactly what let this file
drift out of sync with the real app as it evolved (e.g. it still had
a hardcoded question after final_advisor/pipeline.py was updated to
collect one interactively). Now this file just delegates — there's
exactly one place the combined-app logic lives.

Run:  python main.py
(identical to: python -m final_advisor.pipeline)

To smoke-test one piece at a time instead of the full combined flow:
    python -m economic_report.fetch_data
    python -m economic_report.rag_pipeline
    python -m news_sentiment.pipeline
    python -m investment_advisor.rag_pipeline
"""

from final_advisor.pipeline import (
    collect_investor_profile,
    build_investor_question,
    collect_tickers,
    run_final_advisory,
)

if __name__ == "__main__":
    profile = collect_investor_profile()
    investor_question = build_investor_question(profile)
    tickers = collect_tickers()

    result = run_final_advisory(investor_question, tickers, investor_profile=profile)
    print("\n=== FINAL RECOMMENDATION ===\n")
    print(result)