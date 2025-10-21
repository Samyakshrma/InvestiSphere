import openai
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_EMBEDDING_MODEL

def get_openai_embedding(text: str):
    """Generates an embedding for a given text using Azure OpenAI."""
    client = openai.AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )
    try:
        response = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None
