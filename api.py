import os
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
# Import FileResponse to send files (like PDFs) directly
from fastapi.responses import JSONResponse, FileResponse 

# Project Modules
from data_scraper_agent.scraper import DataScraperAgent
from vector_db.faiss_manager import FAISSManager
from rag.retriever import RAGRetriever
from agents.fundamental_analyst import FundamentalAnalystAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.macroeconomic_agent import MacroeconomicAgent
from cio_agent.cio import CIOAgent
from config import DEFAULT_TICKER

# Global storage for application components
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Initializes all agent instances and managers on startup.
    """
    print("--- FastAPI Startup: Initializing AI System ---")
    
    # 1. Initialize FAISS Manager (DOES NOT load any index yet)
    faiss_manager = FAISSManager()

    # 2. Initialize Core Components (Agents)
    retriever = RAGRetriever(faiss_manager)
    fundamental_analyst = FundamentalAnalystAgent(retriever)
    technical_analyst = TechnicalAnalystAgent() 
    macroeconomic_agent = MacroeconomicAgent()

    # 3. Initialize Orchestrator
    cio_agent = CIOAgent(
        fundamental_analyst=fundamental_analyst,
        technical_analyst=technical_analyst,
        macroeconomic_agent=macroeconomic_agent
    )

    # Store components in application state
    app_state["faiss_manager"] = faiss_manager
    app_state["cio_agent"] = cio_agent
    app_state["technical_analyst"] = technical_analyst 
    
    print("--- Initialization Complete. Server Ready ---")
    yield
    
    # Clean up (optional)
    print("--- FastAPI Shutdown ---")
    app_state.clear()


app = FastAPI(
    title="AI Investment Analyst Backend",
    description="A multi-agent system for financial report generation.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", summary="Root Health Check")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "AI Investment Analyst is running."}


@app.get("/forecast/{ticker}", summary="Get Data for Interactive Forecast Chart")
async def get_forecast_data(ticker: str):
    """
    Retrieves historical data and a 30-day forecast for a given ticker.
    This endpoint is used to populate interactive frontend charts.
    """
    if "technical_analyst" not in app_state:
        raise HTTPException(status_code=503, detail="Technical Analyst not initialized.")

    technical_analyst: TechnicalAnalystAgent = app_state["technical_analyst"]

    try:
        chart_data = technical_analyst.get_chart_data(ticker.upper())
        return JSONResponse(content={
            "ticker": ticker.upper(),
            "timestamp": datetime.datetime.now().isoformat(),
            "chart_data": chart_data
        })
    except Exception as e:
        print(f"Forecast data failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get forecast data: {e}")


@app.get("/analyze/{ticker}", summary="Generate PDF Report (Fast, Data Must Exist)")
async def analyze_ticker_and_download_pdf(ticker: str):
    """
    Generates a full PDF report *only* if the data for the ticker
    has already been ingested. This is the 'fast' endpoint.
    """
    ticker = ticker.upper()
    if "cio_agent" not in app_state or "faiss_manager" not in app_state:
        raise HTTPException(status_code=503, detail="AI System components not initialized.")

    cio_agent: CIOAgent = app_state["cio_agent"]
    faiss_manager: FAISSManager = app_state["faiss_manager"]

    try:
        # 1. Check if local index exists
        is_loaded = faiss_manager.load_index(ticker)
        
        if not is_loaded:
            # 2. If not local, try to download from Azure
            print(f"Local index for {ticker} not found. Attempting sync from Azure...")
            sync_success = faiss_manager.sync_from_azure(ticker)
            
            if not sync_success:
                # 3. If it's not on Azure, the data doesn't exist. Raise 404.
                raise HTTPException(
                    status_code=404, 
                    detail=f"Data for ticker {ticker} not found. Use the POST /generate-and-analyze/{ticker} endpoint to create it."
                )
        
        # 4. Run the pipeline (we are now certain the correct index is loaded)
        pdf_filepath = cio_agent.generate_investment_report(ticker)

        if not pdf_filepath or not os.path.exists(pdf_filepath):
            raise HTTPException(status_code=500, detail="Failed to create PDF report.")

        pdf_filename = os.path.basename(pdf_filepath)
        return FileResponse(
            path=pdf_filepath,
            media_type='application/pdf',
            filename=pdf_filename
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/ingest/{ticker}", summary="Scrape and Ingest New Data (Admin)")
async def ingest_data(ticker: str):
    """
    (Admin) Runs the Data Scraper Agent to gather, embed, and store new data 
    in the ticker-specific FAISS index.
    """
    ticker = ticker.upper()
    faiss_manager: FAISSManager = app_state.get("faiss_manager")

    if not faiss_manager:
         raise HTTPException(status_code=503, detail="FAISS Manager not initialized.")
    
    try:
        data_scraper = DataScraperAgent(faiss_manager)
        data_scraper.scrape_and_process(ticker) 
        
        return JSONResponse(content={
            "status": "success",
            "ticker": ticker,
            "message": f"Data for {ticker} scraped, embedded, and synced to Azure Blob Storage."
        })
    except Exception as e:
        print(f"Data ingestion failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {e}")


# --- NEW ENDPOINT FOR FRONTEND "ONE-CLICK" BUTTON ---
@app.post("/generate-and-analyze/{ticker}", summary="Generate New Report (Slow, All-in-One)")
async def generate_new_report(ticker: str):
    """
    (User Endpoint) This is a slow, all-in-one endpoint that:
    1.  Runs the full data ingestion (scrape, embed, save, sync).
    2.  Runs the full analysis (load index, run agents, create PDF).
    3.  Returns the generated PDF as a download.
    
    WARNING: This endpoint can take 30-60+ seconds and may time out.
    """
    ticker = ticker.upper()
    faiss_manager: FAISSManager = app_state.get("faiss_manager")
    cio_agent: CIOAgent = app_state.get("cio_agent")

    if not faiss_manager or not cio_agent:
         raise HTTPException(status_code=503, detail="System components not initialized.")

    try:
        # --- STEP 1: Run Ingestion ---
        print(f"All-in-One: Starting Step 1 (Ingestion) for {ticker}...")
        data_scraper = DataScraperAgent(faiss_manager)
        data_scraper.scrape_and_process(ticker) 
        print(f"All-in-One: Step 1 complete.")

        # --- STEP 2: Run Analysis ---
        print(f"All-in-One: Starting Step 2 (Analysis) for {ticker}...")
        # (No need to check for index, we *know* it exists now)
        pdf_filepath = cio_agent.generate_investment_report(ticker)

        if not pdf_filepath or not os.path.exists(pdf_filepath):
            raise HTTPException(status_code=500, detail="Failed to create PDF report after ingestion.")
        
        print(f"All-in-One: Step 2 complete. Returning PDF.")
        
        # --- STEP 3: Return PDF File ---
        pdf_filename = os.path.basename(pdf_filepath)
        return FileResponse(
            path=pdf_filepath,
            media_type='application/pdf',
            filename=pdf_filename
        )

    except Exception as e:
        print(f"All-in-One analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Full report generation failed: {e}")