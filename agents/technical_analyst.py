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

    # --- METHOD FOR FORECASTING ENDPOINT ---
    
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
        Performs technical analysis for the given ticker, including forecast interpretation.
        
        Returns:
            tuple: (report_string, chart_file_path)
        """
        print(f"Technical Analyst: Analyzing {ticker}...")
        chart_path = None

        try:
            # 1. Fetch historical data & calculate indicators
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                return f"Technical Analyst: No historical data found for {ticker}.", None

            # Calculate Technical Indicators (SMA 50/200, RSI)
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            delta = hist['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-9)
            hist['RSI'] = 100 - (100 / (1 + rs))

            # 2. INTERPRET FORECAST DIRECTION (NEW LOGIC)
            
            # Use the chart data method to get forecast details
            forecast_data = self.get_chart_data(ticker) 
            
            # Get start/end points of the forecast series
            if forecast_data:
                # Find the latest actual price (end of historical data)
                actual_end_price = next(d['price_actual'] for d in reversed(forecast_data) if d['price_actual'] is not None)
                
                # Find the final predicted price (last point in the list)
                predicted_end_price = forecast_data[-1]['price_forecast']
            else:
                actual_end_price = hist['Close'].iloc[-1]
                predicted_end_price = actual_end_price # Default to flat if forecast failed
            
            # Calculate 30-day percentage change
            forecast_change = ((predicted_end_price - actual_end_price) / actual_end_price) * 100

            if forecast_change > 1.0: # Greater than 1% change is considered upward
                forecast_direction = "Bullish (Upward Trend)"
            elif forecast_change < -1.0: # Less than -1% change is considered downward
                forecast_direction = "Bearish (Downward Trend)"
            else:
                forecast_direction = "Neutral (Flat Trend)"

            # 3. Generate and Save Chart (UNCHANGED)
            chart_path = os.path.join(self.chart_output_dir, f"{ticker}_technical_chart.png")

            ap = [
                mpf.make_addplot(hist['SMA_50'], color='blue', width=0.7),
                mpf.make_addplot(hist['SMA_200'], color='red', width=0.7),
                mpf.make_addplot(hist['RSI'], panel=2, color='purple', ylabel='RSI', ylim=(0,100))
            ]

            mpf.plot(
                hist, type='candle', style='yahoo', title=f"{ticker} 1-Year Technical Analysis",
                ylabel='Price ($)', volume=True, ylabel_lower='Volume', addplot=ap,
                savefig=chart_path, panel_ratios=(4, 1, 1)
            )
            print(f"Chart saved to {chart_path}")
            
            # 4. Use OpenAI to Generate a Summary (PROMPT UPDATED)
            
            latest_data = hist.iloc[-1]
            latest_price = latest_data['Close']
            latest_sma_50 = latest_data['SMA_50']
            latest_sma_200 = latest_data['SMA_200']
            latest_rsi = latest_data['RSI']

            prompt = f"""
            As a technical financial analyst, provide a brief, synthesized summary for {ticker} based on the following data points:
            
            Historical Indicators:
            - Latest Price: ${latest_price:.2f}
            - 50-Day Moving Average (SMA_50): ${latest_sma_50:.2f}
            - 200-Day Moving Average (SMA_200): ${latest_sma_200:.2f}
            - Relative Strength Index (RSI): {latest_rsi:.2f}
            
            **Short-Term Price Forecast (30-Day):**
            - Forecast Trend: **{forecast_direction}** ({forecast_change:.2f}% change projected)

            Based on this data, please interpret:
            1.  The long-term trend (Golden/Death Cross, SMA relationship).
            2.  The current momentum (RSI).
            3.  **The consistency between the historical trend and the short-term forecast.**

            Provide a concise, 2-3 sentence summary of the current technical outlook.
            """

            summary_response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )
            ai_summary = summary_response.choices[0].message.content

            # 5. Formulate the final report
            report = f"""
                        Technical Analysis Report for {ticker}:
                        
                        AI-Generated Interpretation:
                        {ai_summary}

                        Raw Indicator Data (Latest):
                        - Latest Price: ${latest_price:.2f}
                        - 50-Day SMA: {f'${latest_sma_50:.2f}' if not pd.isna(latest_sma_50) else 'N/A'}
                        - 200-Day SMA: {f'${latest_sma_200:.2f}' if not pd.isna(latest_sma_200) else 'N/A'}
                        - 14-Day RSI: {latest_rsi:.2f}
                        - 30-Day Forecast: **{forecast_direction}** ({forecast_change:.2f}%)
                        """
            
            return report, chart_path

        except Exception as e:
            print(f"Technical Analyst: Error during analysis for {ticker} - {e}")
            error_report = f"Technical Analyst: Failed to generate report. Error: {e}"
            return error_report, None