from app.run_pipeline_manager import RAGPipelineManager
verbose = False

# Initialize RAGPipelineManager
rag_manager_semantic = RAGPipelineManager(document_parsing_method="simple",node_parsing_method="semantic", retrieval_type = "fusion_retrieval")
# Create the RAG pipeline
query_pipeline_s = rag_manager_semantic.create_pipeline()