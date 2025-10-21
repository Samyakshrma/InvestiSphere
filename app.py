import argparse
from config import DEFAULT_TICKER
from data_scraper_agent.scraper import DataScraperAgent
from vector_db.faiss_manager import FAISSManager
from rag.retriever import RAGRetriever
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent
from cio_agent.cio import CIOAgent

def main():
    """Main function to run the financial analysis pipeline."""
    parser = argparse.ArgumentParser(description="AI-Powered Financial Analyst")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Stock ticker to analyze (e.g., MSFT, AAPL)")
    parser.add_argument("--scrape", action='store_true', help="Run the data scraper to update data.")
    args = parser.parse_args()

    # --- Phase 1: Foundation & Data Ingestion ---
    faiss_manager = FAISSManager()
    
    # Optional: Scrape new data
    if args.scrape:
        print("Starting data ingestion phase...")
        data_scraper = DataScraperAgent(faiss_manager)
        data_scraper.scrape_and_process(args.ticker)
    else:
        # Ensure we have data locally or from azure
        if not faiss_manager.index:
             print("Local index not found. Attempting to sync from Azure...")
             faiss_manager.sync_from_azure()
             if not faiss_manager.index:
                 print("Could not load index. Please run with --scrape to build one.")
                 return


    # --- Phase 2: Core Analysis ---
    print("\nStarting core analysis phase...")
    
    # Initialize all components
    retriever = RAGRetriever(faiss_manager)
    fundamental_analyst = FundamentalAnalystAgent(retriever)
    technical_analyst = TechnicalAnalystAgent()
    macroeconomic_agent = MacroeconomicAgent()

    # Initialize the orchestrator
    cio = CIOAgent(
        fundamental_analyst=fundamental_analyst,
        technical_analyst=technical_analyst,
        macroeconomic_agent=macroeconomic_agent
    )

    # Generate the final report
    cio.generate_investment_report(args.ticker)


if __name__ == "__main__":
    main()
