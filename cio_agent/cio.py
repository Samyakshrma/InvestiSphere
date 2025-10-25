import openai
import os
import datetime
from fpdf import FPDF
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_CHAT_MODEL

class PDF(FPDF):
    """
    Custom PDF class to handle headers and footers (optional but good practice).
    """
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Investment Analysis Report', 0, 0, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class CIOAgent:
    """
    Chief Investment Officer (CIO) Agent to orchestrate the workflow,
    prompt other agents, and synthesize a final report.
    
    This agent now has its own AI client and is responsible for
    generating the final PDF report.
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
        
        # --- NEW ---
        # Define a directory to save the final PDF reports
        self.report_output_dir = "reports"
        os.makedirs(self.report_output_dir, exist_ok=True)

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
                temperature=0.6,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"CIO Agent: Error during final synthesis - {e}")
            return "Failed to synthesize final report due to an error."

    # --- NEW METHOD ---
    def _create_pdf_report(self, ticker, fundamental_report, technical_report, 
                           chart_path, macroeconomic_report, final_recommendation):
        """
        Assembles all the text and image components into a single PDF file.
        Returns the file path of the generated PDF.
        """
        print(f"CIO Agent: Assembling PDF report for {ticker}...")
        
        pdf = PDF()
        pdf.add_page()
        
        # --- Helper function for writing text ---
        def add_section(title, content):
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, title, 0, 1, 'L')
            pdf.set_font("Arial", size=12)
            # Use multi_cell for long text. Handle potential Unicode errors.
            pdf.multi_cell(0, 5, content.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5) # Add a little space

        # --- 1. Title Page ---
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 15, f"Investment Analysis Report", 0, 1, 'C')
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 10, f"Ticker: {ticker}", 0, 1, 'C')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Generated on: {timestamp}", 0, 1, 'C')
        pdf.ln(10)
        
        # --- 2. Disclaimer ---
        pdf.set_font("Arial", 'I', 8)
        pdf.multi_cell(0, 5, "Disclaimer: This is an AI-generated report and not financial advice. "
                             "All investment decisions should be made with a qualified financial advisor. "
                             "Past performance is not indicative of future results.")
        pdf.ln(10)

        # --- 3. Add Analysis Sections ---
        add_section("1. Fundamental Analysis Summary", fundamental_report)
        add_section("2. Technical Analysis Summary", technical_report)
        
        # --- 4. Add Chart ---
        if chart_path and os.path.exists(chart_path):
            try:
                # Add a title for the chart
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Technical Chart", 0, 1, 'L')
                
                # A4 width is 210mm. Page margins are 10mm each. Usable width = 190mm.
                pdf.image(chart_path, x=10, w=190) 
                pdf.ln(5)
            except Exception as e:
                print(f"CIO Agent: Error embedding chart {chart_path} in PDF. Error: {e}")
                add_section("Technical Chart", "[Error: Chart image could not be embedded.]")
        else:
            add_section("Technical Chart", "[Chart image not available or could not be generated.]")

        # --- 5. Add Macro and Final Sections ---
        add_section("3. Macroeconomic Outlook", macroeconomic_report)
        add_section("Final Recommendation", final_recommendation)

        # --- 6. Save PDF and return path ---
        pdf_filename = f"Investment_Report_{ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_filepath = os.path.join(self.report_output_dir, pdf_filename)
        
        pdf.output(pdf_filepath)
        
        print(f"CIO Agent: PDF report saved to {pdf_filepath}")
        return pdf_filepath


    def generate_investment_report(self, ticker: str):
        """
        Orchestrates the analysis and generates the final PDF report.
        Returns the file path to the generated PDF.
        """
        print(f"\n--- CIO Agent: Generating Investment Report for {ticker} ---")

        # 1. Fundamental Analysis
        fundamental_report = self.fundamental_analyst.analyze(ticker)

        # 2. Technical Analysis
        # --- CHANGE ---
        # Now captures two return values: the text report and the chart file path
        technical_report, chart_path = self.technical_analyst.analyze(ticker)

        # 3. Macroeconomic Analysis (using info from fundamental report as context)
        try:
            company_info_context = fundamental_report.split('**Financial Health Summary**')[0]
        except:
            company_info_context = fundamental_report # fallback

        macroeconomic_report = self.macroeconomic_agent.analyze(ticker, company_info_context)
        
        # 4. Synthesize Final Recommendation (Text)
        final_recommendation = self._synthesize_report(
            fundamental_report, 
            technical_report, 
            macroeconomic_report
        )

        # 5. --- NEW: Create the PDF ---
        # This replaces the old .txt file saving logic
        pdf_filepath = self._create_pdf_report(
            ticker,
            fundamental_report,
            technical_report,
            chart_path, # Pass the chart path
            macroeconomic_report,
            final_recommendation
        )
        
        # --- CHANGE ---
        # Return the path to the PDF, not the text string
        return pdf_filepath