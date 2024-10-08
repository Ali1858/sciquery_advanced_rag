from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config import (QDRANT_URL, QDRANT_API_KEY)


class VectorStoreManager:
    """Manages vector store operations, including creation and loading."""
    
    def __init__(self, vector_db: str = "qdrant"):
        self.vector_db = vector_db

    def prepare_vector_store(self, store_collection_name: str):
        """Prepares and returns a vector store client for the specified collection."""
        if self.vector_db == "qdrant":
            client = QdrantClient(location=QDRANT_URL, api_key=QDRANT_API_KEY)
            async_client = AsyncQdrantClient(location=QDRANT_URL, api_key=QDRANT_API_KEY)
            vector_store = QdrantVectorStore(client=client, aclient=async_client, collection_name=store_collection_name)
        else:
            raise ValueError(f"Vector db: {self.vector_db} not supported. Pick 'qdrant'")
        print(f'{"==="*10} Vector Store created using vector db {self.vector_db}')
        return vector_store

    def create_vector_collection(self, nodes, store_collection_name: str, add_to_existing=False):
        """Creates a vector collection with the specified nodes and collection name."""
        vector_store = self.prepare_vector_store(store_collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        if not add_to_existing:
            VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            vector_store_index.insert_nodes(nodes)
            storage_context.persist()


    def load_vector_collection(self, store_collection_name: str, embed_model):
        """Loads a vector collection with the specified collection name and embedding model."""
        vector_store = self.prepare_vector_store(store_collection_name)
        return VectorStoreIndex.from_vector_store(embed_model=embed_model, vector_store=vector_store)
    

    def add_nodes_to_vector_collection(self, new_nodes, store_collection_name: str):
        """
        Adds new nodes to an existing vector collection.

        :param new_nodes: List of nodes to be added.
        :param store_collection_name: The name of the vector store collection.
        """
        # Prepare the vector store client for the specified collection
        # vector_store = self.prepare_vector_store(store_collection_name)
        
        # Load the existing vector store collection
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        vector_store_index.insert_nodes(new_nodes)
        
        # Persist changes to the vector store
        storage_context.persist()
        print(f"Added {len(new_nodes)} new nodes to the vector collection '{store_collection_name}'.")
