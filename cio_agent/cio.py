import openai
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL

class CIOAgent:
    """
    Chief Investment Officer (CIO) Agent to orchestrate the workflow,
    prompt other agents, and synthesize a final report.
    
    This agent now has its own AI client to perform a final,
    intelligent synthesis of all findings.
    """
    def __init__(self, fundamental_analyst: FundamentalAnalystAgent, 
                 technical_analyst: TechnicalAnalystAgent, 
                 macroeconomic_agent: MacroeconomicAgent):
        self.fundamental_analyst = fundamental_analyst
        self.technical_analyst = technical_analyst
        self.macroeconomic_agent = macroeconomic_agent
        
        # Initialize its own client for the final synthesis
        self.client = openai.AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
        )

    def _synthesize_report(self, fundamental_report, technical_report, macroeconomic_report):
        """
        Uses an LLM to perform a high-level synthesis of the three specialist reports.
        """
        print("CIO Agent: Synthesizing final recommendation...")
        
        prompt = f"""
        As a Chief Investment Officer (CIO), your job is to synthesize the following three specialist reports 
        into a single, cohesive investment recommendation for a client. 

        Do not just list the findings. Explain *how* these factors interact. 
        - Does the technical analysis confirm the fundamental strength?
        - Does the macroeconomic environment support or contradict the company-specific trends?
        - What is the overall, synthesized outlook?

        ---
        Report 1: Fundamental Analysis
        ---
        {fundamental_report}
        
        ---
        Report 2: Technical Analysis
        ---
        {technical_report}

        ---
        Report 3: Macroeconomic Outlook
        ---
        {macroeconomic_report}

        ---
        CIO's Final Synthesized Recommendation:
        (Provide a 2-3 paragraph summary and a clear 'BUY', 'HOLD', or 'SELL' rating.)
        """
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, # Slightly creative but still grounded
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"CIO Agent: Error during final synthesis - {e}")
            return "Failed to synthesize final report due to an error."

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
        # Try to extract just the summary part for a cleaner context
        try:
            company_info_context = fundamental_report.split('**Financial Health Summary**')[0]
        except:
            company_info_context = fundamental_report # fallback

        macroeconomic_report = self.macroeconomic_agent.analyze(ticker, company_info_context)
        
        # 4. Synthesize Final Report
        final_recommendation = self._synthesize_report(
            fundamental_report, 
            technical_report, 
            macroeconomic_report
        )

        final_report_str = f"""
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
        {final_recommendation}

        Disclaimer: This is an AI-generated report and not financial advice.
        """

        #print(final_report_str)
        
        # Save the report to a file
        report_filename = f"investment_report_{ticker}.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report_str)
            
        print(f"\n--- Report for {ticker} saved to {report_filename} ---")
        return final_report_str

