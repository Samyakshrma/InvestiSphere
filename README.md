# InvestiSphere: Your Personal AI Investment Advisor 📈🧠

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Cloud-Azure-0078D4.svg" alt="Cloud">
  <img src="https://img.shields.io/badge/Frontend-Next.js-black.svg" alt="Frontend">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Development-orange.svg" alt="Status">
</p>

<p align="center">
  
</p>

---

## ℹ️ Overview

**InvestiSphere** is a powerful, AI-driven platform designed to provide comprehensive investment analysis reports for publicly traded stocks. Leveraging a sophisticated multi-agent architecture powered by Azure OpenAI, this application gathers real-time financial data, performs fundamental, technical, and macroeconomic analysis, and synthesizes these findings into a downloadable PDF report, complete with charts and forecasts.

This project showcases the integration of Large Language Models (LLMs) with real-time data sources and vector databases to create an intelligent system capable of complex financial reasoning and report generation.

---

## ✨ Key Capabilities

* **Real-time Data Ingestion:** Automatically scrapes company information, news, and historical stock data using `yfinance`.
* **Vector Database Storage:** Processes and embeds scraped data into ticker-specific FAISS vector stores, synchronized with Azure Blob Storage for persistence.
* **Multi-Agent Analysis:**
    * **Fundamental Analyst:** Uses Retrieval-Augmented Generation (RAG) to analyze company health based on scraped data.
    * **Technical Analyst:** Calculates key indicators (SMAs, RSI), generates candlestick charts, provides a 30-day conceptual forecast, and uses AI for interpretation.
    * **Macroeconomic Analyst:** Fetches live market indicators (S&P 500, Treasury Yields, Oil Prices) and uses AI to assess their impact.
    * **Chief Investment Officer (CIO) Agent:** Orchestrates the workflow, synthesizes findings from all specialist agents, and generates the final recommendation.
* **Interactive Frontend:** A modern Next.js dashboard allows users to input tickers, view interactive forecast charts, and trigger report generation.
* **PDF Report Generation:** Creates a professional, downloadable PDF report containing all analysis sections, the technical chart, and the final synthesized recommendation.
* **Asynchronous Processing:** Utilizes FastAPI's `BackgroundTasks` to handle the potentially long-running report generation process without timing out user requests, providing a smooth user experience via polling.

---

## ⚙️ Technologies Used

This project integrates a variety of modern technologies:

* **Backend:**
    * [Python 3.11+](https://www.python.org/): Core programming language.
    * [FastAPI](https://fastapi.tiangolo.com/): High-performance ASGI web framework for building the API.
    * [Uvicorn](https://www.uvicorn.org/): ASGI server to run the FastAPI application.
    * [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service): LLM for text generation, embeddings, and analysis.
    * [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs): Cloud storage for persistent vector database files.
    * [FAISS](https://faiss.ai/): Efficient vector similarity search library from Meta AI.
    * [yfinance](https://pypi.org/project/yfinance/): Library for fetching stock market data.
    * [pandas](https://pandas.pydata.org/): Data manipulation and analysis.
    * [NumPy](https://numpy.org/): Fundamental package for scientific computing.
    * [mplfinance](https://github.com/matplotlib/mplfinance): Financial data visualization (candlestick charts).
    * [fpdf2](https://pyfpdf.github.io/fpdf2/): Library for PDF document generation.
    * [python-dotenv](https://pypi.org/project/python-dotenv/): Loading environment variables.
* **Frontend:**
    * [Next.js](https://nextjs.org/): React framework for building the user interface.
    * [React](https://react.dev/): JavaScript library for building user interfaces.
    * [Tailwind CSS](https://tailwindcss.com/): Utility-first CSS framework for styling.
    * [Chart.js](https://www.chartjs.org/): Library for creating interactive charts.
    * [react-chartjs-2](https://react-chartjs-2.js.org/): React wrapper for Chart.js.

---

## 🏛️ System Architecture

Fin-AI Analyst employs a decoupled, asynchronous, multi-agent architecture designed for scalability and robustness.



**Core Components:**

1.  **Frontend (Next.js):**
    * Provides the user interface for ticker input and displaying results.
    * Makes API calls to the FastAPI backend.
    * Renders the interactive forecast chart using Chart.js.
    * Handles polling for report status and initiates PDF downloads.
2.  **Backend (FastAPI):**
    * Exposes RESTful API endpoints for:
        * Fetching forecast data (`/forecast/{ticker}`).
        * Starting a new report generation job (`/generate-and-analyze/{ticker}`).
        * Checking job status (`/get-report-status/{job_id}`).
        * Downloading the completed PDF report (`/download-report/{job_id}`).
        * (Admin) Manually triggering data ingestion (`/ingest/{ticker}`).
    * Uses `BackgroundTasks` to handle long-running ingestion and analysis processes asynchronously.
    * Manages application state (agent instances, job statuses).
3.  **Data Ingestion Layer:**
    * **Data Scraper Agent:** Triggered by the `/ingest` or `/generate-and-analyze` endpoints. Fetches data from `yfinance`.
    * **Azure OpenAI Embeddings:** Converts textual data (company info, news) into vector embeddings.
    * **FAISS Vector Store:** Stores embeddings and document mappings in **ticker-specific** index files (e.g., `AAPL_index.faiss`). Managed by the `FAISSManager`.
    * **Azure Blob Storage Sync:** The `FAISSManager` synchronizes these index files with Azure Blob Storage for persistence.
4.  **Analysis Layer (Multi-Agent System):**
    * **RAG Retriever:** Fetches relevant documents (company info, news) for a specific ticker from the correct FAISS index file.
    * **Fundamental Analyst Agent:** Uses the RAG context to generate a fundamental health summary via Azure OpenAI.
    * **Technical Analyst Agent:** Fetches live data, calculates indicators, generates a PNG chart, creates forecast data, and uses Azure OpenAI to interpret technical signals *and* the forecast trend.
    * **Macroeconomic Analyst Agent:** Fetches live market indicators, combines them with company context, and uses Azure OpenAI for macro analysis.
5.  **Orchestration & Synthesis Layer:**
    * **CIO Agent:** Manages the workflow. It calls the specialist agents, gathers their findings (including the chart path), synthesizes a final recommendation using Azure OpenAI, and then uses `fpdf2` to assemble the complete PDF report.

**Workflow (User Perspective):**

1.  User enters a ticker (e.g., "AAPL") in the frontend and clicks "Generate Forecast".
2.  Frontend calls `GET /forecast/AAPL`. Backend returns historical + forecast data. Frontend displays the chart.
3.  User clicks "Download Full Report".
4.  Frontend calls `POST /generate-and-analyze/AAPL`.
5.  Backend starts a background task, immediately returns a `job_id`.
6.  Frontend shows a loading message and starts polling `GET /get-report-status/{job_id}` every 5 seconds.
7.  **Background Task:**
    * Runs `DataScraperAgent` for AAPL (fetches, embeds, saves `AAPL_index.faiss`, syncs to Azure). Updates job status to "ingesting".
    * Runs `CIOAgent` orchestration:
        * Loads `AAPL_index.faiss`.
        * Runs Fundamental, Technical, Macro agents.
        * Synthesizes recommendation.
        * Generates `Investment_Report_AAPL_....pdf`. Updates job status to "analyzing".
    * Updates job status to "complete" with the PDF path.
8.  Frontend polling sees "complete" status, enables a download link pointing to `GET /download-report/{job_id}`.
9.  User clicks the link. Backend serves the PDF file via `FileResponse`.

---

## 🚀 Getting Started (Running Locally)

Follow these steps to set up and run the Fin-AI Analyst locally. You will need **two terminals**.

**Prerequisites:**

* [Python](https://www.python.org/) 3.11+ installed.
* [Node.js](https://nodejs.org/) (LTS version recommended) and npm installed.
* An [Azure account](https://azure.microsoft.com/) with access to:
    * Azure OpenAI Service (with models like GPT-4 and text-embedding-ada-002 deployed).
    * Azure Blob Storage.
* Git installed.

**Setup:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Samyakshrma/InvestiSphere.git
    cd InvestiSphere
    ```

2.  **Backend Setup:**
    * Navigate to the backend directory (where `api.py` is located).
    * Create and activate a Python virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # Linux/macOS
        # venv\Scripts\activate    # Windows
        ```
    * Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    * Create a `.env` file in the backend root directory and add your Azure credentials:
        ```dotenv
            # Azure OpenAI Credentials
            AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_KEY
            AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT_URL

            # Azure Blob Storage Credentials
            AZURE_STORAGE_CONNECTION_STRING=YOUR_AZURE_BLOB_CONNECTION_STRING
            AZURE_STORAGE_CONTAINER_NAME=YOUR_CONTAINER_NAME 

# Default Ticker (Optional, used if no ticker provided in older scripts)
DEFAULT_TICKER=MSFT
        ```

3.  **Frontend Setup:**
    * Navigate to the frontend directory (the Next.js project folder).
        ```bash 
        git clone https://github.com/Samyakshrma/InvestiSphere-frontend.git
    * Install Node.js dependencies:
        ```bash
        npm install --legacy-peer-deps
        ```
    * Ensure the `next.config.mjs` (or `.js`) file contains the rewrite rule to proxy `/api` requests to `http://localhost:8000` (see previous instructions if needed).

**Running the Application:**

1.  **Terminal 1: Start the Backend Server:**
    * Make sure your Python virtual environment is activated.
    * Navigate to the backend root directory.
    * Run Uvicorn:
        ```bash
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        ```
    * *Keep this terminal running.*

2.  **Terminal 2: Start the Frontend Dev Server:**
    * Navigate to the frontend project directory.
    * Run the Next.js development server:
        ```bash
        npm run dev
        ```
    * *Keep this terminal running.*

3.  **Access the Application:**
    * Open your web browser and navigate to **`http://localhost:3000`**.

**Using the App:**

1.  Enter a valid stock ticker (e.g., AAPL, MSFT, GOOG) into the sidebar input.
2.  Click "Generate Forecast" to see the interactive chart.
3.  Click "Download Full Report" to start the report generation process.
4.  Wait for the process to complete (the frontend will show status updates).
5.  Once complete, a download button/link will appear. Click it to get your PDF report.

---

## 🗺️ Roadmap / Future Enhancements

* Implement more sophisticated forecasting models (ARIMA, Prophet, LSTM).
* Add user authentication and personalized dashboards.
* Integrate additional data sources (e.g., SEC filings, sentiment analysis).
* Develop more specialized agents (e.g., ESG analyst, options analyst).
* Implement a robust asynchronous task queue (Celery/Redis) for better scalability.
* Add unit and integration tests.
* Deploy to a cloud platform (e.g., Azure App Service, Vercel).

---

## 🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflows. Ensure code adheres to basic linting standards and includes relevant documentation or tests.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

---