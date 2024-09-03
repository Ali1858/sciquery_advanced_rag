from llama_index.core.ingestion import IngestionPipeline
from app.node_parser_factory import NodeParserFactory

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

class IngestionManager:
    """Manages the ingestion process of documents."""
    
    def __init__(self, Settings):
        self.Settings = Settings


    def ingest(self, documents, node_parser_type: str = "semantic", additional_transformation= [], **kwargs):
        print(kwargs)
        """Ingests documents using the specified node parser type."""
        node_parser = NodeParserFactory.get_node_parser(self.Settings.embed_model, parsing_method=node_parser_type, **kwargs)
        pipeline = IngestionPipeline(transformations=[node_parser,self.Settings.embed_model]+additional_transformation)
        return pipeline.run(nodes=documents)
    

    def save_simple_doc_store(self, document_nodes, persist_dir, add_to_existing=False):

        if add_to_existing:
            # Load the existing document store if it exists
            doc_store = self.get_doc_from_store(persist_dir)
        else:
            # Create a new document store
            doc_store = SimpleDocumentStore()

        doc_store.add_documents(document_nodes)
        
        storage_context = StorageContext.from_defaults(docstore=doc_store)
        storage_context.persist(persist_dir)


    def get_doc_from_store(self, persist_dir):
        return SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir)


    def get_storage_contex(self, persist_dir):
        return StorageContext.from_defaults(persist_dir=persist_dir)
