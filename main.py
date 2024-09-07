import os
from fastapi import FastAPI
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


def setup_rag_pipeline():
    validate_rag_params()
    rag_manager = RAGPipelineManager(document_parsing_method=RAG_DOCUMENT_PARSING_METHOD,
                                     node_parsing_method=RAG_NODE_PARSING_METHOD,
                                     retrieval_type=RAG_RETRIEVAL_TYPE)

    # Create the RAG pipeline
    rag_manager.create_query_pipeline()
    return rag_manager


def create_flask_app():
    from app.routes_flask import ManageIndex, QueryIndex
    app = Flask(__name__)
    api = Api(app)
    rag_manager = setup_rag_pipeline()
    app.config['RAG_MANAGER'] = rag_manager
    api.add_resource(QueryIndex, '/api/query')
    api.add_resource(ManageIndex,'/api/documents','/api/documents/<string:file_name>')
    return app


def create_fastapi_app():
    from app.routes_fastapi import app as fastapi_app
    app = FastAPI()
    rag_manager = setup_rag_pipeline()
    fastapi_app.rag_manager = rag_manager
    # Include the routes from routes.py
    app.mount("/", fastapi_app)
    return app


if __name__ == "__main__":
    # Read the backend type from environment variable
    backend_type = os.getenv("BACKEND_TYPE", "flask").lower()
    
    if backend_type == "flask":
        print(f'Launching Backend using Flask API')
        app = create_flask_app()
        app.run(host="0.0.0.0", port=8000)
    else:
        print(f'Launching Backend using Fast API')
        app = create_fastapi_app()
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)