EMBEDDING_MODEL_PATH = "mixedbread-ai/mxbai-embed-large-v1" #'all-MiniLM-L6-v2' # https://huggingface.co/dunzhang/stella_en_400M_v5
RERANKER_MODEL_PATH = "mixedbread-ai/mxbai-rerank-base-v1" 

PDF_DATA_DIR = "data/pdf_documents" 
TOP_K_RETRIEVED = 5
TOP_K_RANKED = 1

OLLAMA_MODEL_NAME = "llama3.1:latest"
QDRANT_URL = "https://f092c1e2-749e-4ec5-918d-ebfbdeb8700b.europe-west3-0.gcp.cloud.qdrant.io"
DEBUG = False
GENERATION_KWARGS = {"temperature":0.85,
                     "top_p": 1.0}

RAG_DOCUMENT_PARSING_METHOD = "simple"
RAG_NODE_PARSING_METHOD = "semantic"
RAG_RETRIEVAL_TYPE = "fusion_retrieval"

import os
qdrant_api_key = os.getenv("QDRANT_API_KEY")

assert qdrant_api_key, "Missing QdrantDB API key, Please set the API key to env variable QDRANT_API_KEY"
QDRANT_API_KEY = qdrant_api_key