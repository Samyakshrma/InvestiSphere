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
        generates embeddings, and stores them.
        """
        print(f"Scraping data for {ticker}...")
        stock = yf.Ticker(ticker)

        # 1. Get company info
        info = stock.info
        processed_info = f"Company: {info.get('longName')}, Sector: {info.get('sector')}, Industry: {info.get('industry')}, Summary: {info.get('longBusinessSummary')}"

        # 2. Get recent news
        news = stock.news
        
        # --- FIX IS HERE ---
        # Changed 'title' to 'headline' and 'publisher' to 'link'.
        # Used .get() for safety to prevent future KeyErrors if a key is missing.
        processed_news = [f"Headline: {item.get('headline', 'No Headline')}. Link: {item.get('link', 'No Link')}" for item in news[:5]] # Get top 5 news

        # Combine data into documents
        documents = [processed_info] + processed_news
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = [get_openai_embedding(doc) for doc in documents]
        
        # Filter out any None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        valid_documents = [doc for doc, emb in zip(documents, embeddings) if emb is not None]

        if not valid_embeddings:
            print("Could not generate any embeddings.")
            return

        # Add to FAISS index
        self.faiss_manager.add_to_index(valid_embeddings, valid_documents)
        print(f"Data for {ticker} scraped, processed, and stored in FAISS index.")
        
        # Sync with Azure
        self.faiss_manager.sync_to_azure()

