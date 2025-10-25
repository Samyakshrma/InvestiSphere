import os
import openai
import yfinance as yf
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load .env variables first
load_dotenv()

# Import our project modules
import config
from utils import get_openai_embedding

def check_config():
    """1. Test if all .env variables are loaded."""
    print("--- 1. Testing Configuration (.env variables) ---")
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_STORAGE_CONTAINER_NAME"
    ]
    
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"❌ FAILURE: Missing required environment variables: {', '.join(missing)}")
        return False
    
    print("✅ SUCCESS: All key environment variables are loaded.")
    return True

def test_yfinance():
    """2. Test connection to yfinance API."""
    print("\n--- 2. Testing yfinance API ---")
    try:
        ticker = yf.Ticker("MSFT")
        info = ticker.info
        if 'longName' in info and info['longName'] == "Microsoft Corporation":
            print("✅ SUCCESS: yfinance API connection is working.")
            return True
        else:
            print("❌ FAILURE: yfinance API returned unexpected data.")
            return False
    except Exception as e:
        print(f"❌ FAILURE: Could not connect to yfinance. Error: {e}")
        return False

def test_openai_embedding():
    """3. Test Azure OpenAI for Embeddings."""
    print("\n--- 3. Testing Azure OpenAI (Embeddings) ---")
    try:
        embedding = get_openai_embedding("This is a connectivity test")
        if embedding and len(embedding) > 0:
            print(f"✅ SUCCESS: Azure OpenAI (Embeddings) is working. (Got vector of dim {len(embedding)})")
            return True
        else:
            print("❌ FAILURE: Azure OpenAI (Embeddings) call returned no data.")
            return False
    except Exception as e:
        print(f"❌ FAILURE: Azure OpenAI (Embeddings) call raised an error.")
        print(f"   Error: {e}")
        print("   (Check your AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and OPENAI_EMBEDDING_MODEL name)")
        return False

def test_openai_chat():
    """4. Test Azure OpenAI for Chat Completions."""
    print("\n--- 4. Testing Azure OpenAI (Chat) ---")
    try:
        client = openai.AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": "Say 'test'."}],
            temperature=0,
            max_tokens=5
        )
        if response.choices[0].message.content:
            print("✅ SUCCESS: Azure OpenAI (Chat) is working.")
            return True
        else:
            print("❌ FAILURE: Azure OpenAI (Chat) call returned no content.")
            return False
    except Exception as e:
        print(f"❌ FAILURE: Azure OpenAI (Chat) call raised an error.")
        print(f"   Error: {e}")
        print("   (Check your AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and OPENAI_CHAT_MODEL name)")
        return False

def test_azure_blob():
    """5. Test Azure Blob Storage connection."""
    print("\n--- 5. Testing Azure Blob Storage ---")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(config.AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(config.AZURE_STORAGE_CONTAINER_NAME)
        
        # Try to list blobs (v12+ compatible)
        list(container_client.list_blobs())  # <- remove max_results
        
        print("✅ SUCCESS: Azure Blob Storage connection is OK.")
        return True
    except Exception as e:
        print(f"❌ FAILURE: Could not connect to Azure Blob Storage.")
        print(f"   Error: {e}")
        print("   (Check your AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME)")
        return False


def main():
    """Runs all connectivity tests."""
    print("=====================================")
    print("--- Starting Connectivity Test ---")
    print("=====================================")
    
    # Run all tests and track results
    results = [
        check_config(),
        test_yfinance(),
        test_openai_embedding(),
        test_openai_chat(),
        test_azure_blob()
    ]
    
    print("\n=====================================")
    print("--- Test Summary ---")
    print("=====================================")
    
    if all(results):
        print("✅✅✅ All external connections are working correctly! ✅✅✅")
    else:
        print("❌❌❌ One or more tests failed. Please review the output above. ❌❌❌")

if __name__ == "__main__":
    main()
