import openai
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL
from rag.retriever import RAGRetriever

class FundamentalAnalystAgent:
    """
    Analyzes the financial health of a company using context from the RAG module.
    This version uses multiple, specific queries to ensure all document types
    (summary + news) are retrieved.
    """

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )

    def analyze(self, ticker: str):
        """
        Performs fundamental analysis and generates a summary.
        """
        print("Fundamental Analyst: Analyzing...")
        
        # --- FIX: Use multiple, specific queries ---
        
        # 1. Query for the company summary
        summary_query = f"Company business summary and financial health for {ticker}"
        summary_context = self.retriever.retrieve(summary_query, k=1) # Get the top 1 summary doc
        
        # 2. Query for recent news
        news_query = f"Recent news headlines for {ticker}"
        news_context = self.retriever.retrieve(news_query, k=4) # Get the top 4 news docs
        
        # 3. Combine the context
        combined_context = (
            "--- Company Summary Context ---\n"
            f"{summary_context}\n\n"
            "--- Recent News Context ---\n"
            f"{news_context}"
        )

        if not summary_context and not news_context:
            return "Fundamental Analyst: Could not retrieve relevant context."

        # The prompt remains the same, as it was already fixed.
        prompt = f"""
        As a financial analyst, your task is to provide a summary of {ticker}'s financial health.
        Base your analysis *only* on the context provided below. Do not use any outside knowledge.

        ---
        Provided Context:
        {combined_context}
        ---

        Analysis Task:
        Based *only* on the context above:
        1.  Summarize the company's business, sector, and financial health from the 'Company Summary Context'.
        2.  Extract any *recent news headlines* listed in the 'Recent News Context'. If that section is empty or irrelevant, state that "No recent news was found in the provided data."
        
        Provide a concise, well-structured analysis.
        """

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, 
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Fundamental Analyst: Error during analysis - {e}"

