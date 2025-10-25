import os
import yfinance as yf
import pandas as pd
import mplfinance as mpf  # For plotting financial charts
import openai
import random # Added for simple forecast modeling
from datetime import timedelta # Added for date manipulation
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL

class TechnicalAnalystAgent:
    """
    Analyzes and visualizes technical indicators for a given stock ticker.
    
    This agent fetches historical data, calculates key indicators (Moving Averages, RSI),
    generates a chart, and uses an LLM to interpret the findings.
    
    It also provides a method to get raw data for interactive frontend charts.
    """

    def __init__(self, chart_output_dir="charts"):
        """
        Initializes the agent.
        
        Args:
            chart_output_dir (str): The directory where chart images will be saved.
        """
        self.chart_output_dir = chart_output_dir
        os.makedirs(self.chart_output_dir, exist_ok=True)
        
        # Initialize the Azure OpenAI client for generating analysis
        self.client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )

    # --- NEW METHOD FOR FORECASTING ENDPOINT ---
    
    def get_chart_data(self, ticker: str):
        """
        Fetches historical data and generates a simple 30-day forecast.
        This method is designed to provide raw data for an interactive frontend chart.
        
        Returns:
            list: A list of dictionaries, each containing 'date', 'price_actual', 
                  and 'price_forecast'.
        """
        print(f"Technical Analyst: Fetching chart data for {ticker}...")
        try:
            # 1. Fetch 1 year of historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                raise ValueError(f"No historical data found for {ticker}.")

            # 2. Format historical data
            chart_data = []
            for date, row in hist.iterrows():
                chart_data.append({
                    "date": date.strftime('%Y-%m-%d'), # Format date as string
                    "price_actual": row['Close'],
                    "price_forecast": None # No forecast for historical data
                })

            # 3. Generate a simple 30-day conceptual forecast
            # This is a placeholder for a real model (like ARIMA or Prophet)
            # We'll use a simple random walk with a slight drift.
            last_price = hist['Close'].iloc[-1]
            last_date = hist.index[-1]
            
            # Calculate a simple daily drift (avg daily change over last quarter)
            daily_returns = hist['Close'].pct_change().tail(60) # Last 60 trading days
            avg_daily_drift = daily_returns.mean()
            std_dev = daily_returns.std() # Volatility

            forecast_price = last_price
            for i in range(1, 31): # Forecast for 30 days
                # Create a random shock based on volatility, add the drift
                daily_shock = random.gauss(avg_daily_drift, std_dev)
                forecast_price *= (1 + daily_shock)
                
                # Get the next business day (approximate)
                next_date = last_date + timedelta(days=i)
                
                chart_data.append({
                    "date": next_date.strftime('%Y-%m-%d'),
                    "price_actual": None, # No actual price for future dates
                    "price_forecast": forecast_price
                })
                
            return chart_data

        except Exception as e:
            print(f"Technical Analyst: Error getting chart data for {ticker} - {e}")
            raise e # Re-raise exception to be caught by FastAPI

    # --- EXISTING METHOD FOR REPORT GENERATION ---

    def analyze(self, ticker: str):
        """
        Performs technical analysis for the given ticker.
        
        1. Fetches 1 year of historical data.
        2. Calculates SMA 50, SMA 200, and RSI 14.
        3. Generates and saves a chart.
        4. Uses OpenAI to generate a summary based on the latest data.
        
        Returns:
            str: A formatted report including the AI summary and chart path.
        """
        print(f"Technical Analyst: Analyzing {ticker}...")

        try:
            # 1. Fetch historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                return f"Technical Analyst: No historical data found for {ticker}."

            # 2. Calculate Technical Indicators
            
            # Simple Moving Averages (SMA)
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

            # Relative Strength Index (RSI) - 14-day is standard
            delta = hist['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Use exponential moving average for RSI calculation for smoother results
            avg_gain = gain.ewm(com=13, adjust=False).mean() # com = 13 is equivalent to alpha = 1/14
            avg_loss = loss.ewm(com=13, adjust=False).mean()

            # Handle potential division by zero if avg_loss is 0
            rs = avg_gain / avg_loss.replace(0, 1e-9) # replace 0 with a very small number
            hist['RSI'] = 100 - (100 / (1 + rs))

            # 3. Generate and Save Chart
            chart_path = os.path.join(self.chart_output_dir, f"{ticker}_technical_chart.png")

            ap = [
                mpf.make_addplot(hist['SMA_50'], color='blue', width=0.7),
                mpf.make_addplot(hist['SMA_200'], color='red', width=0.7),
                mpf.make_addplot(hist['RSI'], panel=2, color='purple', ylabel='RSI', ylim=(0,100)) # panel=2
            ]

            mpf.plot(
                hist,
                type='candle',
                style='yahoo',
                title=f"{ticker} 1-Year Technical Analysis",
                ylabel='Price ($)',
                volume=True,           # panel 1
                ylabel_lower='Volume',
                addplot=ap,
                savefig=chart_path,
                panel_ratios=(4, 1, 1) # (Price, Volume, RSI)
            )
            print(f"Chart saved to {chart_path}")
            
            # 4. Use OpenAI to Generate a Summary
            
            # Get the latest data points for the prompt
            latest_data = hist.iloc[-1]
            latest_price = latest_data['Close']
            latest_sma_50 = latest_data['SMA_50']
            latest_sma_200 = latest_data['SMA_200']
            latest_rsi = latest_data['RSI']

            prompt = f"""
            As a technical financial analyst, provide a brief summary for {ticker} based on the following latest data points. 
            The full chart has already been generated and saved.

            - Latest Price: ${latest_price:.2f}
            - 50-Day Moving Average (SMA_50): ${latest_sma_50:.2f}
            - 200-Day Moving Average (SMA_200): ${latest_sma_200:.2f}
            - Relative Strength Index (RSI): {latest_rsi:.2f}

            Based on this data, please interpret:
            1.  The price's position relative to the 50-day and 200-day MAs.
            2.  The relationship between the 50-day and 200-day MAs (e.g., is it a "Golden Cross" or "Death Cross" pattern?).
            3.  The RSI level (e.g., overbought (>70), oversold (<30), or neutral).

            Provide a concise, 2-3 sentence summary of the current technical outlook.
            """

            summary_response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )
            ai_summary = summary_response.choices[0].message.content

            # 5. Formulate the final report
            report = f"""
                        Technical Analysis Report for {ticker}:
                        Chart: A new chart has been generated and saved to: {chart_path}

                        AI-Generated Interpretation:
                        {ai_summary}

                        Raw Indicator Data (Latest):
                        - Latest Price: ${latest_price:.2f}
                        - 50-Day SMA: {f'${latest_sma_50:.2f}' if not pd.isna(latest_sma_50) else 'N/A (insufficient data)'}
                        - 200-Day SMA: {f'${latest_sma_200:.2f}' if not pd.isna(latest_sma_200) else 'N/A (insufficient data)'}
                        - 14-Day RSI: {latest_rsi:.2f}
                        """
            return report

        except Exception as e:
            print(f"Technical Analyst: Error during analysis for {ticker} - {e}")
            return f"Technical Analyst: Failed to generate report. Error: {e}"