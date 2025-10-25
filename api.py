import os
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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
    Initializes the entire AI system (FAISS, Agents) on startup.
    """
    print("--- FastAPI Startup: Initializing AI System ---")
    
    # 1. Initialize FAISS Manager (Loads from local/Azure)
    faiss_manager = FAISSManager()
    
    # If index is empty, try to sync from Azure
    if not faiss_manager.index:
         print("Local index not found. Attempting to sync from Azure...")
         try:
             faiss_manager.sync_from_azure()
         except Exception as e:
             print(f"WARNING: Could not sync from Azure. Data ingestion needed. Error: {e}")

    # 2. Initialize Core Components
    retriever = RAGRetriever(faiss_manager)
    fundamental_analyst = FundamentalAnalystAgent(retriever)
    technical_analyst = TechnicalAnalystAgent() # This agent is now needed by two endpoints
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
    # --- CHANGE ---
    # Store the technical analyst so the /forecast endpoint can access it
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


# --- NEW ENDPOINT ---
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
        # Call the new method from the technical agent
        chart_data = technical_analyst.get_chart_data(ticker.upper())
        
        return JSONResponse(content={
            "ticker": ticker.upper(),
            "timestamp": datetime.datetime.now().isoformat(),
            "chart_data": chart_data
        })

    except Exception as e:
        print(f"Forecast data failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get forecast data: {e}")


@app.get("/analyze/{ticker}", summary="Generate Investment Analysis Report")
async def analyze_ticker(ticker: str):
    """
    Triggers the CIO Agent to generate a full investment analysis report for a given ticker.
    """
    # Ensure all components were loaded at startup
    if "cio_agent" not in app_state:
        raise HTTPException(status_code=503, detail="AI System components not initialized.")

    cio_agent: CIOAgent = app_state["cio_agent"]

    try:
        # Run the full orchestration pipeline
        report_str = cio_agent.generate_investment_report(ticker.upper())
        
        return JSONResponse(content={
            "ticker": ticker.upper(),
            "timestamp": datetime.datetime.now().isoformat(),
            "report": report_str
        })

    except Exception as e:
        print(f"Analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/ingest/{ticker}", summary="Scrape and Ingest New Data")
async def ingest_data(ticker: str):
    """
    Runs the Data Scraper Agent to gather, embed, and store new data in the FAISS index.
    """
    faiss_manager: FAISSManager = app_state.get("faiss_manager")

    if not faiss_manager:
         raise HTTPException(status_code=503, detail="FAISS Manager not initialized.")
    
    try:
        data_scraper = DataScraperAgent(faiss_manager)
        data_scraper.scrape_and_process(ticker.upper())
        
        return JSONResponse(content={
            "status": "success",
            "ticker": ticker.upper(),
            "message": "Data scraped, embedded, and synced to Azure Blob Storage."
        })
    except Exception as e:
        print(f"Data ingestion failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {e}")