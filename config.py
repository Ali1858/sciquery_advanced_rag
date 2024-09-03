
# https://huggingface.co/dunzhang/stella_en_400M_v5
# dunzhang/stella_en_400M_v5
# Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {query}

EMBEDDING_MODEL_PATH = "mixedbread-ai/mxbai-embed-large-v1" #'all-MiniLM-L6-v2'
RERANKER_MODEL_PATH = "mixedbread-ai/mxbai-rerank-base-v1"

PDF_DATA_DIR = "data/pdf_documents" 
TOP_K_RETRIEVED = 5
TOP_K_RANKED = 1

OLLAMA_MODEL_NAME = "llama3.1:latest"
QDRANT_URL = ""
QDRANT_API_KEY = ""

DEBUG = False

INDEX_AND_BIB_DIR = "data/index_and_bib"
INDEX_ARRAY_NAME = "index.npy"
BIB_JSON_NAME = "bibliographies.json"
DEVICE = "MPS" 
GENERATION_KWARGS = {"temperature":0.85,
                     "top_p": 1.0}
MAX_TOKENS = 4096
