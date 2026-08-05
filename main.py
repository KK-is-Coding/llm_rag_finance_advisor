"""
main.py
=======
Single entry point for the whole project.

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