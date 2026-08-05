"""
final_advisor/pipeline.py
==============================
Combines all three sub-projects into one interactive financial advisor:

  1. investment_advisor  -> what similar PAST investors chose, given a
                             profile collected FROM THE CURRENT USER
                             (peer-based advice, for reference only)
  2. economic_report      -> current financial data for 1-3 tickers
                             the user names
  3. news_sentiment       -> recent news sentiment for those same tickers

...then asks the LLM to synthesize everything into one recommendation,
including a side-by-side comparison when more than one ticker is given.

Depends on:
    - economic_report/fetch_data.py
    - economic_report/rag_pipeline.py
    - investment_advisor/rag_pipeline.py
    - news_sentiment/pipeline.py
    - shared/llm.py
"""

from economic_report.fetch_data import fetch_and_save
from economic_report.rag_pipeline import run_report_query
from investment_advisor.rag_pipeline import run_advisor_query
from news_sentiment.pipeline import run_sentiment_summary
from shared.llm import get_llm

MAX_TICKERS = 3

SYNTHESIS_TEMPLATE = """You are a financial advisor. Combine the research below into
one clear, well-organized recommendation for the investor.

--- THE INVESTOR'S ACTUAL STATED PROFILE (these are facts the user told us) ---
{stated_profile}

--- PEER-BASED REFERENCE ADVICE (retrieved from OTHER, PAST investors with a
similar profile in our database — for context only; these are NOT the
current investor's own attributes, do not present any detail from this
section as if the current investor stated it) ---
{investor_advice}

--- COMPANY RESEARCH ---
{ticker_context}

{instruction}

The investor listed these preferred investment avenues: {avenue}. Address
those avenues directly (not just the named stock tickers) somewhere in your
answer — e.g. whether mutual funds, equity, fixed deposits, and/or government
bonds fit their goals, using the peer-based reference advice above as context.

Keep it concise and use a table for any figures, including a side-by-side
comparison table if more than one company is discussed. Only state facts
about the investor that came from "THE INVESTOR'S ACTUAL STATED PROFILE" —
never invent or restate peer data as if it belongs to the current investor."""


# ---------------------------------------------------------------------------
# Step 0: collecting the investor's profile and tickers interactively.
# These questions mirror the exact fields investment_advisor/data_prep.py
# uses to build its training documents (age, gender, Avenue, Purpose,
# Duration) so the generated question retrieves well against that data.
# ---------------------------------------------------------------------------

def collect_investor_profile() -> dict:
    """Ask the user for the profile fields the investment_advisor RAG expects."""
    print("Let's build your investor profile.\n")
    age = input("Your age: ").strip()
    gender = input("Gender (Male/Female): ").strip()
    avenue = input(
        "Preferred investment avenue "
        "(e.g. Mutual Fund, Equity, Fixed Deposits, Government Bonds): "
    ).strip()
    purpose = input(
        "Purpose of investing "
        "(e.g. Wealth Creation, Retirement Plan, Rainy Days, Educational Requirements): "
    ).strip()
    duration = input(
        "Investment duration "
        "(e.g. Less than 1 year, 1-3 years, 3-5 years, More than 5 years): "
    ).strip()
    return {"age": age, "gender": gender, "Avenue": avenue, "Purpose": purpose, "Duration": duration}


def build_investor_question(profile: dict) -> str:
    """Turning a collected profile into the natural-language question the RAG chain expects."""
    return (
        f"I'm a {profile['age']}-year-old {profile['gender']} looking to invest "
        f"in {profile['Avenue']} for {profile['Purpose']} over the next "
        f"{profile['Duration']}. What are my options?"
    )


def format_profile_block(profile: dict) -> str:
    """Rendering the user's own stated profile as a clean, unambiguous fact list."""
    return (
        f"- Age: {profile.get('age', 'not specified')}\n"
        f"- Gender: {profile.get('gender', 'not specified')}\n"
        f"- Preferred investment avenue(s): {profile.get('Avenue', 'not specified')}\n"
        f"- Purpose: {profile.get('Purpose', 'not specified')}\n"
        f"- Investment duration: {profile.get('Duration', 'not specified')}\n"
        f"(Nothing else about this investor was provided — expected returns, "
        f"risk tolerance, monitoring frequency, etc. were NOT stated by them.)"
    )


def collect_tickers() -> list[str]:
    """Ask the user for 1-3 tickers to research and compare."""
    raw = input(
        f"\nWhich stock ticker(s) do you want researched/compared? "
        f"(comma-separated, up to {MAX_TICKERS}, e.g. MSFT,AAPL,GOOGL): "
    ).strip()
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")
    if len(tickers) > MAX_TICKERS:
        print(f"Only comparing the first {MAX_TICKERS} tickers you entered.")
        tickers = tickers[:MAX_TICKERS]
    return tickers


# ---------------------------------------------------------------------------
# Step 1-3: gather research per ticker, then synthesize.
# ---------------------------------------------------------------------------

def gather_ticker_research(ticker: str, exchange: str = "US") -> dict:
    """Fetch current financials + news sentiment for one ticker."""
    fetch_and_save(ticker=ticker, exchange=exchange)
    financial_report = run_report_query(f"{ticker} Financial Report")
    news_summary = run_sentiment_summary(f"{ticker} News")
    return {
        "ticker": ticker,
        "financial_report": financial_report,
        "news_summary": news_summary,
    }


def run_final_advisory(
    investor_question: str,
    tickers: list[str],
    investor_profile: dict = None,
    exchange: str = "US",
) -> str:
    """
    Full pipeline: peer-based reference advice + live financials + news
    sentiment for 1-3 tickers -> one synthesized (and, if >1 ticker,
    comparative) recommendation grounded in the investor's ACTUAL stated
    profile.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")
    if len(tickers) > MAX_TICKERS:
        raise ValueError(f"Pass at most {MAX_TICKERS} tickers, got {len(tickers)}.")

    investor_profile = investor_profile or {}
    stated_profile = format_profile_block(investor_profile)
    avenue = investor_profile.get("Avenue", "not specified")

    print("[1/3] Getting peer-based reference advice...")
    advisor_result = run_advisor_query(investor_question)
    investor_advice = advisor_result["result"]

    ticker_sections = []
    failed_tickers = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"[2/3] ({i}/{len(tickers)}) Researching {ticker}...")
        try:
            research = gather_ticker_research(ticker, exchange=exchange)
            ticker_sections.append(
                f"--- {ticker} ---\n"
                f"Financial data:\n{research['financial_report']}\n\n"
                f"News sentiment:\n{research['news_summary']}\n"
            )
        except Exception as e:
            print(f"    WARNING: couldn't get data for {ticker}, skipping it. ({e})")
            failed_tickers.append(ticker)
            ticker_sections.append(
                f"--- {ticker} ---\n"
                f"(No data available for this ticker — data fetch failed.)\n"
            )
    if failed_tickers:
        print(f"    Note: continuing without data for: {', '.join(failed_tickers)}")
    combined_ticker_context = "\n\n".join(ticker_sections)

    if len(tickers) > 1:
        instruction = (
            f"Compare {', '.join(tickers)} against each other. For EACH ticker "
            f"individually, give a short verdict (2-3 sentences) on whether it "
            f"suits this investor's stated goals — every ticker listed must get "
            f"its own verdict, not just the top pick. Then end with an overall "
            f"recommendation naming which one (or which combination) best fits "
            f"this investor, and why."
        )
    else:
        instruction = (
            f"Explain what this investor should consider about {tickers[0]} "
            f"specifically, based on the financial data and news sentiment above."
        )

    print("[3/3] Synthesizing final recommendation...")
    prompt = SYNTHESIS_TEMPLATE.format(
        stated_profile=stated_profile,
        investor_advice=investor_advice,
        ticker_context=combined_ticker_context,
        instruction=instruction,
        avenue=avenue,
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    profile = collect_investor_profile()
    investor_question = build_investor_question(profile)
    tickers = collect_tickers()

    result = run_final_advisory(investor_question, tickers, investor_profile=profile)
    print("\n=== FINAL RECOMMENDATION ===\n")
    print(result)