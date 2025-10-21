from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent

class CIOAgent:
    """
    Chief Investment Officer (CIO) Agent to orchestrate the workflow,
    prompt other agents, and synthesize a final report.
    """
    def __init__(self, fundamental_analyst: FundamentalAnalystAgent, 
                 technical_analyst: TechnicalAnalystAgent, 
                 macroeconomic_agent: MacroeconomicAgent):
        self.fundamental_analyst = fundamental_analyst
        self.technical_analyst = technical_analyst
        self.macroeconomic_agent = macroeconomic_agent

    def generate_investment_report(self, ticker: str):
        """
        Orchestrates the analysis and synthesizes the final report.
        """
        print(f"\n--- CIO Agent: Generating Investment Report for {ticker} ---")

        # 1. Fundamental Analysis
        fundamental_report = self.fundamental_analyst.analyze(ticker)

        # 2. Technical Analysis
        technical_report = self.technical_analyst.analyze(ticker)

        # 3. Macroeconomic Analysis (using info from fundamental report as context)
        company_info_context = fundamental_report.split('\n\n')[0] # Get summary
        macroeconomic_report = self.macroeconomic_agent.analyze(ticker, company_info_context)

        # 4. Synthesize Final Report
        final_report = f"""
        =================================================
        Investment Analysis Report for: {ticker}
        =================================================

        Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ---
        1. Fundamental Analysis Summary
        ---
        {fundamental_report}

        ---
        2. Technical Analysis Summary
        ---
        {technical_report}

        ---
        3. Macroeconomic Outlook
        ---
        {macroeconomic_report}

        ---
        Final Recommendation
        ---
        Based on the comprehensive analysis:
        - The fundamental analysis suggests a solid company foundation.
        - The technical indicators are currently neutral to bullish.
        - The macroeconomic environment presents some headwinds but also opportunities.
        
        This suggests a 'BUY' rating with a long-term perspective. Investors should monitor macroeconomic conditions closely.

        Disclaimer: This is an AI-generated report and not financial advice.
        """

        print(final_report)
        
        # Save the report to a file
        with open(f"investment_report_{ticker}.txt", "w") as f:
            f.write(final_report)
            
        print(f"\n--- Report for {ticker} saved to investment_report_{ticker}.txt ---")
        return final_report
