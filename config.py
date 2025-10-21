import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = "2023-05-15"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "gpt-4"

# Azure Blob Storage Credentials
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = "financial-data"

# Data Source (using Alpha Vantage as an example)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# Using yfinance as a free alternative for this example
DEFAULT_TICKER = "MSFT"

# FAISS Index
FAISS_INDEX_PATH = "faiss_index"
INDEX_FILE_NAME = "index.faiss"
INDEX_MAPPING_FILE_NAME = "index_mapping.pkl"
