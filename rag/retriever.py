from utils import get_openai_embedding
from vector_db.faiss_manager import FAISSManager

class RAGRetriever:
    """
    Retrieval-Augmented Generation module to retrieve relevant
    documents from the vector DB based on a query.
    """

    def __init__(self, faiss_manager: FAISSManager):
        self.faiss_manager = faiss_manager

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieves the top k most relevant documents for a given query.
        """
        query_embedding = get_openai_embedding(query)
        if query_embedding is None:
            return "Could not generate embedding for the query."

        results = self.faiss_manager.search(query_embedding, k=k)
        
        # Format the context
        context = "\n---\n".join([doc for doc, score in results])
        return context
