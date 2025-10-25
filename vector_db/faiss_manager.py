import os
import faiss
import numpy as np
import pickle
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, FAISS_INDEX_PATH, INDEX_FILE_NAME, INDEX_MAPPING_FILE_NAME

class FAISSManager:
    """
    Manages FAISS indices and their synchronization with Azure Blob Storage.
    
    The manager is now refactored to handle separate, ticker-specific index files 
    to prevent data corruption.
    """

    def __init__(self):
        # Only create the base directory; index files are managed dynamically.
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        self.index = None
        self.doc_mapping = {}

    def _get_paths(self, ticker: str):
        """Generates the local and Azure-friendly paths for a specific ticker."""
        base_name = ticker.upper()
        
        # Local paths (e.g., FAISS_INDEX_PATH/NVDA_index.faiss)
        index_file = f"{base_name}_{INDEX_FILE_NAME}"
        mapping_file = f"{base_name}_{INDEX_MAPPING_FILE_NAME}"
        
        local_index_path = os.path.join(FAISS_INDEX_PATH, index_file)
        local_mapping_path = os.path.join(FAISS_INDEX_PATH, mapping_file)
        
        return local_index_path, local_mapping_path, index_file, mapping_file

    def load_index(self, ticker: str):
        """Loads the FAISS index and mapping for a specific ticker from local files."""
        local_index_path, local_mapping_path, _, _ = self._get_paths(ticker)
        
        if os.path.exists(local_index_path) and os.path.exists(local_mapping_path):
            self.index = faiss.read_index(local_index_path)
            with open(local_mapping_path, 'rb') as f:
                self.doc_mapping = pickle.load(f)
            print(f"FAISS index for {ticker} loaded from local files.")
            return True
        
        self.index = None
        self.doc_mapping = {}
        return False

    def save_index(self, ticker: str):
        """Saves the current FAISS index and mapping to ticker-specific local files."""
        local_index_path, local_mapping_path, _, _ = self._get_paths(ticker)
        
        if self.index:
            faiss.write_index(self.index, local_index_path)
            with open(local_mapping_path, 'wb') as f:
                pickle.dump(self.doc_mapping, f)
            print(f"FAISS index for {ticker} saved locally.")

    def create_index(self, embeddings, documents):
        """Creates a new FAISS index based on the provided data."""
        if not embeddings:
            print("No embeddings provided to create index.")
            return

        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.doc_mapping = {i: doc for i, doc in enumerate(documents)}

    def add_to_index(self, ticker: str, new_embeddings, new_documents):
        """Adds new embeddings and documents to an existing index for the given ticker."""
        
        # 1. Ensure the correct index is loaded (or create a new one)
        self.load_index(ticker)
        
        if self.index is None:
            self.create_index(new_embeddings, new_documents)
        else:
            # 2. Add new data
            self.index.add(np.array(new_embeddings, dtype=np.float32))
            
            # Update mapping
            start_index = len(self.doc_mapping)
            for i, doc in enumerate(new_documents):
                self.doc_mapping[start_index + i] = doc
                
        # 3. Save the updated index locally
        self.save_index(ticker)

    def search(self, ticker: str, query_embedding, k=5):
        """Searches the index for a specific ticker for the top k similar documents."""
        
        # Ensure the correct index is loaded before searching
        if not self.load_index(ticker):
             # The index should have been loaded before the search is called,
             # but we check again for robustness.
             return []
             
        if self.index is None:
            return []
            
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [(self.doc_mapping[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def sync_to_azure(self, ticker: str):
        """Uploads the local index files for a specific ticker to Azure Blob Storage."""
        local_index_path, local_mapping_path, index_file, mapping_file = self._get_paths(ticker)

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        
        for local_path, azure_name in [(local_index_path, index_file), (local_mapping_path, mapping_file)]:
            if os.path.exists(local_path):
                with open(local_path, "rb") as data:
                    container_client.upload_blob(name=azure_name, data=data, overwrite=True)
                print(f"Uploaded {azure_name} to Azure Blob Storage.")

    def sync_from_azure(self, ticker: str):
        """Downloads the index files for a specific ticker from Azure Blob Storage."""
        local_index_path, local_mapping_path, index_file, mapping_file = self._get_paths(ticker)

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

        success = True
        for local_path, azure_name in [(local_index_path, index_file), (local_mapping_path, mapping_file)]:
             try:
                 with open(local_path, "wb") as download_file:
                    download_file.write(container_client.download_blob(azure_name).readall())
                 print(f"Downloaded {azure_name} from Azure Blob Storage.")
             except Exception as e:
                 # It's okay if a file doesn't exist on Azure for a new ticker
                 print(f"WARNING: Could not download {azure_name} from Azure (Error: {e})")
                 success = False
                 
        if success:
             self.load_index(ticker)
        return success