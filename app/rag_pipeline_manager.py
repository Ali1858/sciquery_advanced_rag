import os

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import StorageContext
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
                 retrieval_type = "",
                 name_suffix = ""):
        
        self.DOCUMENT_DIRECTORY = document_directory
        self.DOC_STORE_DIRECTORY = doc_store_directory
        self.model_context_window = 4096
        self.model_max_new_tokens = 786
        self.name_suffix = name_suffix

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

        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.ingestion_manager = IngestionManager(Settings)
        self.query_pipeline_builder = QueryPipelineBuilder(Settings, node_parsing_method=self.node_parsing_method)

        self.doc_store_name = self.doc_store_name_template.format(name=self.node_parsing_method, suffix=self.name_suffix)
        self.collection_name = self.vectore_db_col_name_template.format(name=self.node_parsing_method, suffix=self.name_suffix)

        try:
            doc_store_name = self.doc_store_name_template.format(name=self.node_parsing_method, suffix=self.name_suffix)
            self.ingestion_manager.get_doc_from_store(os.path.join(self.DOC_STORE_DIRECTORY,doc_store_name))

            self.run_injes = False
            self.save_vector = False
            self.parse_doc_again =  False
        except Exception as e:
            print(e)
            self.run_injes = True
            self.save_vector = True
            self.parse_doc_again =  True
        print(f'pipeline will parse document {self.parse_doc_again}, save vector {self.save_vector} and run injestion {self.run_injes}.')


    def run_ingestion_pipeline(self, run_pipeline, documents, add_to_existing=False):
        """
        Runs the ingestion pipeline and stores documents in a document store.

        Args:
            run_pipeline (bool): Whether to run the pipeline.
            node_parser_type (str): Type of node parser to use.
            documents (List[Document]): Documents to be ingested.

        Returns:
            List[Document]: A list of documents retrieved from the document store.
        """
        if run_pipeline and len(documents) > 0:
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

            self.ingestion_manager.save_simple_doc_store(nodes, os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name), add_to_existing)

        return self.ingestion_manager.get_doc_from_store(os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name))


    def save_and_load_vector_store(self, save_vector_store, add_to_existing=False, nodes=None):
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

            self.vector_store_manager.create_vector_collection(saving_nodes, self.collection_name,add_to_existing=add_to_existing)

        return self.vector_store_manager.load_vector_collection(self.collection_name, embed_model=Settings.embed_model)
    

    def update_for_one_doc(self, pdf_file, **kwargs):
        # Step 1: Prepare documents
        new_docs = self.document_processor.prepare_single_document( pdf_file, method=self.document_parsing_method)
        self.reload_pipeline(run_and_save=True, add_to_existing = True, docs=new_docs, **kwargs)

    
    def load_existing_doc_store(self):
        # Step 1: Load the existing document store
        persist_dir = os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name)
        return self.ingestion_manager.get_doc_from_store(persist_dir)


    def delete_doc_from_index(self,document_name):
        persist_dir = os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name)
        doc_store = self.load_existing_doc_store()
        # Step 2: Find nodes related to the specified document name
        nodes_to_delete = [node_id for node_id, node in doc_store.docs.items() if node.metadata.get("file_name") == document_name]

        if not nodes_to_delete:
            print(f"No nodes found for document name: {document_name}")
            return False
        else:
            print(f'number of to be deleted: {len(nodes_to_delete)}')
            # Step 3: Remove nodes from the document store
            for node_id in nodes_to_delete:
                doc_store.delete_document(node_id)

            # Step 4: Persist changes to the document store
            storage_context = StorageContext.from_defaults(docstore=doc_store)
            storage_context.persist(persist_dir)

            self.vector_store_manager.load_vector_collection(self.collection_name, embed_model=Settings.embed_model)

            # Step 5: Prepare vector store
            vector_store = self.vector_store_manager.prepare_vector_store(self.collection_name)
            # Step 6: Remove vectors from the vector store
            vector_store.delete_nodes(nodes_to_delete)
            self.reload_pipeline()
            return True


    def reload_pipeline(self,run_and_save=False, add_to_existing = False, docs=[], **kwargs):

        # Run ingestion pipeline
        self.doc_store = self.run_ingestion_pipeline(run_pipeline=run_and_save, documents=docs, add_to_existing=add_to_existing)
        nodes = list(self.doc_store.docs.values())
        print(f'total number of nodes found in doc store: {len(nodes)}')

        # Save and load vector store
        self.vector_index = self.save_and_load_vector_store(save_vector_store=run_and_save, nodes=nodes, add_to_existing=add_to_existing)

        # Build query pipeline
        doc_store_path = os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name)
        self.query_pipeline = self.query_pipeline_builder.build_query_pipeline(self.retrieval_type,
                                                                          base_index=self.vector_index,
                                                                          doc_store = self.doc_store,
                                                                          storage_context = self.ingestion_manager.get_storage_contex(doc_store_path),
                                                                          **kwargs)
    

    def create_query_pipeline(self, **kwargs):
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
        self.doc_store = self.run_ingestion_pipeline(run_pipeline=self.run_injes, documents=documents)
        nodes = list(self.doc_store.docs.values())
    
        # Step 3: Save and load vector store
        self.vector_index = self.save_and_load_vector_store(save_vector_store=self.save_vector, nodes=nodes)

        # Step 4: Build query pipeline
        doc_store_path = os.path.join(self.DOC_STORE_DIRECTORY,self.doc_store_name)
        self.query_pipeline = self.query_pipeline_builder.build_query_pipeline(self.retrieval_type,
                                                                          base_index=self.vector_index,
                                                                          doc_store = self.doc_store,
                                                                          storage_context = self.ingestion_manager.get_storage_contex(doc_store_path),
                                                                          **kwargs)


    def run_query_pipeline(self,query):
        output, steps = self.query_pipeline.run_with_intermediates(topic=query)
        answer = output.message.content

        relevant_nodes = []
        for node in steps["retriever"].outputs["output"]:
            metadata = node.metadata
            relevant_nodes.append(
                {"text" : node.text,
                 "node_id" : node.id_,
                 "page_num" : metadata["page_label"],
                 "file_name" : metadata["file_name"]
                 }
                 )
        return steps, {"relevant_nodes":relevant_nodes, "query":query, "answer": answer}