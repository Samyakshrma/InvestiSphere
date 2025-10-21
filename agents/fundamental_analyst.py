import openai
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL
from rag.retriever import RAGRetriever

class FundamentalAnalystAgent:
    """
    Analyzes the financial health of a company using context from the RAG module.
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
        query = f"Provide a summary of the financial health of {ticker}."
        context = self.retriever.retrieve(query)

        if not context:
            return "Fundamental Analyst: Could not retrieve relevant context."

        prompt = f"""
        As a financial analyst, analyze the following information about {ticker} and provide a summary of its financial health.
        Focus on its business summary, sector, and any recent news.

        Context:
        {context}

        Analysis:
        """

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Fundamental Analyst: Error during analysis - {e}"
