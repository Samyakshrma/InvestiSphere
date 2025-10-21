import openai
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL

class MacroeconomicAgent:
    """
    Analyzes macroeconomic trends relevant to a given company/sector.
    """
    def __init__(self):
        self.client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )

    def analyze(self, ticker: str, company_info: str):
        """
        Analyzes and summarizes market trends.
        """
        print("Macroeconomic Agent: Analyzing...")

        prompt = f"""
        As a macroeconomic analyst, provide an analysis of the current market trends that could impact {ticker},
        given the following company information: {company_info}.

        Consider factors like inflation, interest rates, and overall market sentiment.
        
        Analysis:
        """
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Macroeconomic Agent: Error during analysis - {e}"
