import os
import faiss
import numpy as np
import pickle
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, FAISS_INDEX_PATH, INDEX_FILE_NAME, INDEX_MAPPING_FILE_NAME

class FAISSManager:
    """Manages a FAISS index and its synchronization with Azure Blob Storage."""

    def __init__(self):
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        self.index_path = os.path.join(FAISS_INDEX_PATH, INDEX_FILE_NAME)
        self.mapping_path = os.path.join(FAISS_INDEX_PATH, INDEX_MAPPING_FILE_NAME)
        self.index = None
        self.doc_mapping = {}
        self.load_from_local()

    def create_index(self, embeddings, documents):
        """Creates a new FAISS index."""
        if not embeddings:
            print("No embeddings provided to create index.")
            return

        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.doc_mapping = {i: doc for i, doc in enumerate(documents)}
        self.save_to_local()

    def add_to_index(self, new_embeddings, new_documents):
        """Adds new embeddings and documents to an existing index."""
        if self.index is None:
            self.create_index(new_embeddings, new_documents)
        else:
            self.index.add(np.array(new_embeddings, dtype=np.float32))
            # Update mapping
            start_index = len(self.doc_mapping)
            for i, doc in enumerate(new_documents):
                self.doc_mapping[start_index + i] = doc
        self.save_to_local()

    def search(self, query_embedding, k=5):
        """Searches the index for the top k similar documents."""
        if self.index is None:
            return []
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [(self.doc_mapping[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save_to_local(self):
        """Saves the FAISS index and mapping to local files."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(self.doc_mapping, f)
            print("FAISS index saved locally.")

    def load_from_local(self):
        """Loads the FAISS index and mapping from local files."""
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, 'rb') as f:
                self.doc_mapping = pickle.load(f)
            print("FAISS index loaded from local files.")

    def sync_to_azure(self):
        """Uploads the local index files to Azure Blob Storage."""
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        
        for file_name in [INDEX_FILE_NAME, INDEX_MAPPING_FILE_NAME]:
            local_path = os.path.join(FAISS_INDEX_PATH, file_name)
            if os.path.exists(local_path):
                with open(local_path, "rb") as data:
                    container_client.upload_blob(name=file_name, data=data, overwrite=True)
                print(f"Uploaded {file_name} to Azure Blob Storage.")

    def sync_from_azure(self):
        """Downloads the index files from Azure Blob Storage."""
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

        for file_name in [INDEX_FILE_NAME, INDEX_MAPPING_FILE_NAME]:
             with open(os.path.join(FAISS_INDEX_PATH, file_name), "wb") as download_file:
                download_file.write(container_client.download_blob(file_name).readall())
             print(f"Downloaded {file_name} from Azure Blob Storage.")
        self.load_from_local()
