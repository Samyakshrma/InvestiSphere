import os
import argparse
from dotenv import load_dotenv

# Import the agents you want to test
from agents.technical_analyst import TechnicalAnalystAgent
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent
# Import other agents as you create them

def setup_environment():
    """Loads environment variables and checks for critical ones."""
    print("--- 1. Setting up Environment ---")
    load_dotenv()
    
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("❌ FAILURE: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set.")
        print("Please check your .env file.")
        return False
    
    print("Environment variables loaded successfully.\n")
    return True

def test_technical_agent(ticker: str):
    """Runs a test on the TechnicalAnalystAgent."""
    print(f"--- 2a. Testing TechnicalAnalystAgent for {ticker} ---")
    try:
        agent = TechnicalAnalystAgent(chart_output_dir="charts")
        report = agent.analyze(ticker)
        
        print("\n--- Technical Analyst Report ---")
        print(report)
        print("----------------------------------\n")
        
        # Verify chart file creation
        expected_chart_path = os.path.join("charts", f"{ticker}_technical_chart.png")
        if os.path.exists(expected_chart_path):
            print(f"✅ SUCCESS: Technical Analyst test passed. Chart created at {expected_chart_path}")
            return True
        else:
            print(f"❌ FAILURE: Technical Analyst test failed. Chart not found at {expected_chart_path}")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: TechnicalAnalystAgent failed. Error: {e}")
        return False

def test_fundamental_agent(ticker: str, retriever=None):
    """Runs a test on the FundamentalAnalystAgent."""
    print(f"--- 2b. Testing FundamentalAnalystAgent for {ticker} ---")
    print("NOTE: This test requires a valid, pre-scraped FAISS index for the ticker.")
    
    # This test is more complex as it needs a retriever.
    # For a simple isolation test, we can't easily run it without
    # the scraper and FAISS index. We will stub it for now.
    
    # TODO: To properly test this, you'd initialize a RAGRetriever
    # pointing to a test index.
    
    if not retriever:
        print("WARNING: No retriever provided. Mocking FundamentalAnalystAgent run.")
        # We can't run this agent without a retriever, which needs a DB.
        # So we'll just initialize it to check for errors.
        try:
            agent = FundamentalAnalystAgent(retriever=None) # Pass None to test initialization
            print("Agent initialized (mock test)...")
            # We can't call .analyze() without a retriever
            # report = agent.analyze(ticker) 
            print("✅ SUCCESS: Fundamental Analyst (mock test) passed.")
            return True
        except Exception as e:
            print(f"❌ FAILURE: FundamentalAnalystAgent initialization failed. Error: {e}")
            return False
    
    # If a retriever was passed, you could run the full test:
    # try:
    #     agent = FundamentalAnalystAgent(retriever=retriever)
    #     report = agent.analyze(ticker)
    #     print("\n--- Fundamental Analyst Report ---")
    #     print(report)
    #     print("----------------------------------\n")
    #     print("✅ SUCCESS: Fundamental Analyst test passed.")
    #     return True
    # except Exception as e:
    #     print(f"❌ FAILURE: FundamentalAnalystAgent failed. Error: {e}")
    #     return False

def test_macro_agent():
    """Runs a test on the MacroeconomicAgent."""
    print(f"--- 2c. Testing MacroeconomicAgent ---")
    try:
        agent = MacroeconomicAgent()
        report = agent.analyze()
        
        print("\n--- Macroeconomic Report ---")
        print(report)
        print("------------------------------\n")
        
        if report and "Macroeconomic Analysis Report" in report:
            print(f"✅ SUCCESS: Macroeconomic Agent test passed.")
            return True
        else:
            print(f"❌ FAILURE: Macroeconomic Agent test failed. Report was empty or invalid.")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: MacroeconomicAgent failed. Error: {e}")
        return False

def main(ticker: str):
    """
    Main test runner.
    Define your test pipeline here.
    """
    if not setup_environment():
        return

    print(f"--- 2. Starting Agent Tests for Ticker: {ticker} ---")
    
    # Define all the test jobs you want to run
    # This is where you can easily add or remove agents from the test
    test_jobs = [
        {"name": "Technical Analyst", "func": test_technical_agent, "args": [ticker]},
        {"name": "Fundamental Analyst", "func": test_fundamental_agent, "args": [ticker]},
        {"name": "Macroeconomic Agent", "func": test_macro_agent, "args": []},
    ]

    results = {}
    
    for job in test_jobs:
        print(f"\n--- Running Job: {job['name']} ---")
        test_passed = job["func"](*job["args"])
        results[job['name']] = "✅ PASSED" if test_passed else "❌ FAILED"
    
    # 4. Print final summary
    print("\n=========================")
    print("--- Test Run Summary ---")
    print("=========================")
    for name, result in results.items():
        print(f"- {name}: {result}")
    print("=========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test financial agents in isolation.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to test (e.g., AAPL, MSFT)")
    args = parser.parse_args()
    
    main(args.ticker)
