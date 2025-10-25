from utils import get_openai_embedding
from vector_db.faiss_manager import FAISSManager

class RAGRetriever:
    """
    Retrieval-Augmented Generation module to retrieve relevant
    documents from the vector DB based on a query.
    
    This version relies on the FAISSManager to load a ticker-specific
    index, removing the need for post-retrieval filtering.
    """

    def __init__(self, faiss_manager: FAISSManager):
        self.faiss_manager = faiss_manager

    # --- FIX: Updated method signature ---
    # It now accepts 'ticker' as an explicit argument from the agent.
    def retrieve(self, ticker: str, query: str, k: int = 5):
        """
        Retrieves the top k most relevant documents for a given query
        from the specified ticker's index.
        
        Args:
            ticker (str): The specific ticker index to search (e.g., "AAPL", "NVDA").
            query (str): The search query (e.g., "Recent news headlines...").
            k (int): The number of documents to retrieve.
            
        Returns:
            str: A formatted string containing the combined context.
        """
        
        # --- REMOVED ---
        # No need to guess the ticker from the query anymore.
        # ticker = query.split()[0].upper() 
        
        query_embedding = get_openai_embedding(query)
        if query_embedding is None:
            return "Could not generate embedding for the query."

        # 2. Search the Ticker-Specific Index
        # The FAISSManager handles loading the correct index file (e.g., NVDA_index.faiss)
        # based on the ticker, eliminating the risk of data corruption from other tickers.
        results = self.faiss_manager.search(ticker, query_embedding, k=k)
        
        if not results:
            return f"No context found for ticker: {ticker}. Index may be empty or not yet scraped."

        # 3. Format the context (No need to filter anymore!)
        context = "\n---\n".join([doc for doc, score in results])
        
        return context