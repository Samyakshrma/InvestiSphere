import yfinance as yf
from utils import get_openai_embedding
from vector_db.faiss_manager import FAISSManager

class DataScraperAgent:
    """
    Agent to gather, process, and embed data from yfinance
    and store it in the vector DB.
    """

    def __init__(self, faiss_manager: FAISSManager):
        self.faiss_manager = faiss_manager

    def scrape_and_process(self, ticker: str):
        """
        Scrapes financial data for a given ticker, processes it,
        generates embeddings, and stores them in the ticker-specific index.
        
        RAISES:
            Exception: If data cannot be retrieved or embeddings cannot be generated.
        """
        ticker = ticker.upper() # Ensure ticker is uppercase for consistency
        print(f"Scraping data for {ticker}...")
        stock = yf.Ticker(ticker)

        # 1. Get company info
        info = stock.info
        
        # --- FIX 1: ADD VALIDATION CHECK ---
        # Check if yfinance returned valid data. If not info or no longName, it's a bad ticker or network error.
        if not info or 'longName' not in info or info.get('longName') is None:
            error_msg = f"Failed to retrieve valid company info for {ticker}. The ticker may be invalid or the data source (yfinance) is unavailable."
            print(f"DataScraperAgent ERROR: {error_msg}")
            raise Exception(error_msg) # Fail loudly
            
        # Prepend the ticker to the document for easier identification if needed.
        processed_info = f"{ticker}: Company: {info.get('longName')}, Sector: {info.get('sector')}, Industry: {info.get('industry')}, Summary: {info.get('longBusinessSummary')}"

        # 2. Get recent news
        news = stock.news
        
        # Format news headlines
        processed_news = [f"{ticker}: Headline: {item.get('headline', 'No Headline')}. Link: {item.get('link', 'No Link')}" for item in news[:5]] 

        # Combine data into documents
        documents = [processed_info] + processed_news
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = [get_openai_embedding(doc) for doc in documents]
        
        # Filter out any None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        valid_documents = [doc for doc, emb in zip(documents, embeddings) if emb is not None]

        # --- FIX 2: CHANGE SILENT RETURN TO LOUD EXCEPTION ---
        if not valid_embeddings:
            error_msg = f"Could not generate any valid embeddings for {ticker}. This may be due to an OpenAI API issue or empty source data."
            print(f"DataScraperAgent ERROR: {error_msg}")
            raise Exception(error_msg) # Fail loudly

        # Add to FAISS index
        self.faiss_manager.add_to_index(ticker, valid_embeddings, valid_documents)
        print(f"Data for {ticker} scraped, processed, and stored in FAISS index.")
        
        # Sync with Azure
        self.faiss_manager.sync_to_azure(ticker)