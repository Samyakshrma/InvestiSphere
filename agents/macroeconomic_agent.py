import openai
import yfinance as yf
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL

class MacroeconomicAgent:
    """
    Analyzes macroeconomic trends relevant to a given company/sector.
    This agent is "grounded" by fetching live data for key economic indicators
    before prompting the LLM.
    """
    def __init__(self):
        self.client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )
        # Define key macroeconomic indicators
        self.macro_tickers = {
            "S&P 500": "^GSPC",
            "10-Yr Treasury": "^TNX",
            "Crude Oil": "CL=F"
        }

    def _get_macro_context(self):
        """
        Fetches live data and recent news for key macro indicators.
        """
        print("Macroeconomic Agent: Fetching live macro context...")
        context_string = "--- Real-Time Macroeconomic Context ---\n"
        
        for name, ticker_str in self.macro_tickers.items():
            ticker = yf.Ticker(ticker_str)
            info = ticker.info
            hist = ticker.history(period="5d") # Get last 5 days
            
            # Get latest price and previous close
            latest_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = latest_price - prev_close
            change_percent = (change / prev_close) * 100

            context_string += f"\nIndicator: {name} ({ticker_str})\n"
            context_string += f"  Latest Value: {latest_price:.2f}\n"
            context_string += f"  Change: {change:+.2f} ({change_percent:+.2f}%)\n"
            
            # Get latest news
            news = ticker.news[:2] # Get top 2 news items
            if news:
                context_string += "  Recent News:\n"
                for item in news:
                    context_string += f"    - {item.get('headline', 'No Headline')}\n"
        
        context_string += "--- End of Context ---\n"
        return context_string

    def analyze(self, ticker: str, company_info: str):
        """
        Analyzes and summarizes market trends using grounded, real-time data.
        """
        print("Macroeconomic Agent: Analyzing...")
        
        # 1. Get the live macro context
        context_string = self._get_macro_context()

        # 2. Create the grounded prompt
        prompt = f"""
        As a macroeconomic analyst, provide an analysis of the current market trends.
        Your analysis MUST be based *only* on the real-time context and company info provided below.
        Do NOT use any outdated, general knowledge.

        {context_string}

        Company to Analyze: {ticker}
        Company Info: {company_info}

        ---
        Analysis Task:
        Based *only* on the context above, analyze the current market trends.
        Specifically address:
        1.  **Overall Market Sentiment:** (Use S&P 500 data)
        2.  **Interest Rates & Inflation:** (Use 10-Yr Treasury and Crude Oil data)
        3.  **Potential Impact:** Briefly state how these factors might impact {ticker}.
        
        Provide a concise analysis.
        """
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, # Lower temperature for more factual response
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Macroeconomic Agent: Error during analysis - {e}"