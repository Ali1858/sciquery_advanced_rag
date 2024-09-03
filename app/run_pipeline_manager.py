import os

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

from app.vector_store_manager import VectorStoreManager
from app.document_processor import DocumentProcessor
from app.node_parser_factory import NodeParserFactory
from app.ingestion_manager import IngestionManager
from app.query_pipeline_builder import QueryPipelineBuilder
from config import (OLLAMA_MODEL_NAME,
                    DEBUG,
                    EMBEDDING_MODEL_PATH,
                    PDF_DATA_DIR)


class RAGPipelineManager:
    """
    RAGPipelineManager orchestrates the entire process of setting up and running a RAG pipeline.
    It uses VectorStoreManager, DocumentProcessor, NodeParserFactory, IngestionManager, and QueryPipelineBuilder.
    """
    def __init__(self, 
                 document_directory = PDF_DATA_DIR,
                 doc_store_directory = "data",
                 model_provider = "ollama",
                 embedding_provider = "huggingface",
                 vector_db="qdrant",
                 document_parsing_method = "manual_parsing",
                 node_parsing_method = "",
                 retrieval_type = ""):
        
        self.DOCUMENT_DIRECTORY = document_directory
        self.DOC_STORE_DIRECTORY = doc_store_directory
        self.model_context_window = 4096
        self.model_max_new_tokens = 786

        # Vector DB, Model and Embeddingprovider
        self.vector_db = vector_db
        self.model_provider = model_provider
        self.embedding_provider = embedding_provider

        self.doc_store_name_template = "{name}-document-store{suffix}"
        self.vectore_db_col_name_template = "{name}-embed-collection{suffix}"

        # Chunking method and Retrieval type
        self.node_parsing_method = node_parsing_method
        self.retrieval_type = retrieval_type
        self.document_parsing_method = document_parsing_method
        
        # Set up the service context
        self.service_context = self.setup_service_context()    


        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.ingestion_manager = IngestionManager(Settings)
        self.query_pipeline_builder = QueryPipelineBuilder(Settings)
        

    def __get_llms__ (self, model_provider):
        DEBUG = False
        if model_provider == "ollama":
            model_name = OLLAMA_MODEL_NAME
            llm = Ollama(model=OLLAMA_MODEL_NAME,request_timeout=120.0)
        else:
            raise ValueError (f"Model provider : {model_provider} not supported. Pick Ollama")
        
        print(f'{"==="*10} LLM {model_name} is loaded successfully using the provider {model_provider}')
        if DEBUG:
            print('Testing the LLM output for query: Who is Paul Graham?')
            print(llm.complete("Who is Paul Graham?"))
        return llm


    def __get_embedding_model__ (self, embedding_provider):
        DEBUG = False
        if embedding_provider=="huggingface":
            embed_model_name = EMBEDDING_MODEL_PATH
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH)
        else:
            raise ValueError (f"Embedding provider : {embedding_provider} not supported. Pick 'huggingface")
        
        print(f'{"==="*10} Embedding {embed_model_name} is loaded successfully using the provider {embedding_provider}')
        if DEBUG:
            print('Testing the Embedding output for query: Hellow World"')
            embeddings = embed_model.get_text_embedding("Hello World!")
            print(len(embeddings))
            print(embeddings[:5])
        return embed_model    
    

    def setup_service_context(self):
        Settings.llm = self.__get_llms__(self.model_provider)
        Settings.embed_model = self.__get_embedding_model__(self.embedding_provider)

        self.run_injes = self.should_run_pipeline()
        self.save_vector = self.should_save_vector()
        self.parse_doc_again = self.should_parse_document()

        if self.parse_doc_again:
            self.run_injes = True
            self.save_vector = True


    def should_run_pipeline(self):
        return True


    def should_save_vector(self):
        return True
    

    def should_parse_document(self):
        return True


    def run_ingestion_pipeline(self, run_pipeline, documents, suffix):
        """
        Runs the ingestion pipeline and stores documents in a document store.

        Args:
            run_pipeline (bool): Whether to run the pipeline.
            node_parser_type (str): Type of node parser to use.
            documents (List[Document]): Documents to be ingested.

        Returns:
            List[Document]: A list of documents retrieved from the document store.
        """
        if run_pipeline:
            if self.node_parsing_method == "semantic":
                nodes = self.ingestion_manager.ingest(documents, node_parser_type="semantic")

            elif self.node_parsing_method == "sentence_window":
                nodes = self.ingestion_manager.ingest(
                    documents,
                    node_parser_type="sentence_window",
                    window_size=3,
                    window_metadata_key="window",
                    original_text_metadata_key="original_sentence"
                )

            elif self.node_parsing_method == "hierarchical":
                nodes = self.ingestion_manager.ingest(
                    documents,
                    node_parser_type="hierarchical",
                    chunk_sizes=[2048, 512, 128]
                )

            self.doc_store_name = self.doc_store_name_template.format(name=self.node_parsing_method, suffix=suffix)
            self.ingestion_manager.save_simple_doc_store(nodes, os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name))

        return self.ingestion_manager.get_doc_from_store(os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name))


    def save_and_load_vector_store(self, save_vector_store, suffix, nodes=None):
        """
        Saves nodes to a vector store and loads them for retrieval.

        Args:
            save_vector_store (bool): Whether to save the vector store.
            node_parser_type (str): Type of node parser to use.
            nodes (List[Node]): Nodes to be saved.

        Returns:
            VectorStoreIndex: The loaded vector store index.
        """
        if save_vector_store and nodes is not None:
            if self.node_parsing_method == "semantic":
                saving_nodes = nodes
                print(f'Total nodes: {len(nodes)}')

            elif self.node_parsing_method == "sentence_window":
                saving_nodes = nodes
                print(f'Total nodes: {len(nodes)}')

            elif self.node_parsing_method == "hierarchical":
                leaf_nodes = get_leaf_nodes(nodes)
                root_nodes = get_root_nodes(nodes)
                print(f'Total nodes: {len(nodes)}, leaf nodes: {len(leaf_nodes)}, root nodes: {len(root_nodes)}')
                saving_nodes = leaf_nodes

            self.collection_name = self.vectore_db_col_name_template.format(name=self.node_parsing_method, suffix=suffix)
            self.vector_store_manager.create_vector_collection(saving_nodes, self.collection_name)

        return self.vector_store_manager.load_vector_collection(self.collection_name, embed_model=Settings.embed_model)
    

    def create_pipeline(self, **kwargs):
        """
        Sets up the entire RAG pipeline, from document preparation to query pipeline creation.

        Args:
            pdf_paths (str): Path to the directory containing PDF files.
            store_collection_name (str): Name of the collection for the vector store.
            node_parser_type (str): Method to use for parsing documents ('semantic', 'simple', etc.).
            rag_type (str): Type of RAG pipeline ('simple', 'fusion_retrieval', 'small_to_big').
            **kwargs: Additional arguments for specific parsers and retrievers.

        Returns:
            QueryPipeline: A configured query pipeline ready to run queries.
        """
        # Step 1: Prepare documents
        documents = self.document_processor.prepare_documents( self.DOCUMENT_DIRECTORY, parse_doc_again=self.parse_doc_again, method=self.document_parsing_method)

        # Step 2: Run ingestion pipeline
        self.doc_store = self.run_ingestion_pipeline(run_pipeline=self.run_injes, documents=documents, suffix=kwargs.get('suffix',''))
        nodes = list(self.doc_store.docs.values())
    
        # Step 3: Save and load vector store
        self.vector_index = self.save_and_load_vector_store(save_vector_store=self.save_vector, nodes=nodes, suffix=kwargs.get('suffix',''))

        # Step 4: Build query pipeline
        doc_store_path = os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name)
        query_pipeline = self.query_pipeline_builder.build_query_pipeline(self.retrieval_type,
                                                                          base_index=self.vector_index,
                                                                          doc_store = self.doc_store,
                                                                          storage_context = self.ingestion_manager.get_storage_contex(doc_store_path),
                                                                          **kwargs)

        return query_pipeline
    

    # def persist_index(self):
    #     """
    #     Persist the current state of the index to storage.
    #     """
    #     self.index.storage_context.persist(self.INDEX_DIRECTORY)
    
    # def reload_index(self):
    #     """
    #     Reload the index from persistent storage.
    #     """
    #     self.index = self.load_index()
    #     self.index = self.vector_store_manager.load_vector_collection(collection_name, embed_model=Settings.embed_model)
    
    # def get_index(self):
    #     """
    #     Retrieve the current index.
    #     """
    #     return self.index
    
    # def delete_index(self, ref_document_id):
    #     """
    #     Delete document Index.

    #     Args:
    #         ref_document_id (str): Reference document id of the document.
    #     """
    #     # Delete the document index based on ref_doc_id
    #     self.index.delete_ref_doc(ref_doc_id=ref_document_id, delete_from_docstore=True)
    #     # Store the updated index
    #     self.persist_index()
    #     # Reload the index
    #     self.reload_index()