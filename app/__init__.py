from flask import Flask
from flask_restful import Api

from app.rag_pipeline_manager import RAGPipelineManager
from config import RAG_DOCUMENT_PARSING_METHOD, RAG_NODE_PARSING_METHOD, RAG_RETRIEVAL_TYPE


def validate_rag_params():
    message = f"Retrieval type {RAG_RETRIEVAL_TYPE} not supported with the node parsing method {RAG_NODE_PARSING_METHOD}, choose different parser"
    if RAG_RETRIEVAL_TYPE == "small_to_big": # "simple", "fusion_retrieval", "small_to_big"
        assert RAG_NODE_PARSING_METHOD in ("sentence_window", "hierarchical"), message # "semantic", "sentence_window",  "hierarchical"
    elif RAG_RETRIEVAL_TYPE == "fusion_retrieval" or RAG_RETRIEVAL_TYPE == "simple": # "simple", "fusion_retrieval", "small_to_big"
        assert RAG_NODE_PARSING_METHOD == "semantic", message # "semantic", "sentence_window",  "hierarchical"
    else:
        raise ValueError(f"Retrieval type {RAG_RETRIEVAL_TYPE} not supported")


def setup_rag_pipeline(app):
    # Initialize RAGPipelineManager
    validate_rag_params()
    rag_manager = RAGPipelineManager(document_parsing_method=RAG_DOCUMENT_PARSING_METHOD,
                                              node_parsing_method=RAG_NODE_PARSING_METHOD,
                                              retrieval_type = RAG_RETRIEVAL_TYPE)
    
    # Create the RAG pipeline
    rag_manager.create_query_pipeline()
    app.config['RAG_MANAGER'] = rag_manager


def create_app():
    app = Flask(__name__)
    api = Api(app)
    setup_rag_pipeline(app)

    from app.routes import initialize_routes
    initialize_routes(api)
    return app
