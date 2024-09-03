import os
from glob import glob
import re
from typing import List
from types import MethodType 
from functools import partial

from qdrant_client import QdrantClient, AsyncQdrantClient

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.query_pipeline import InputComponent, FunctionComponent, QueryPipeline
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever

from llama_index.core.node_parser import (SemanticSplitterNodeParser,
                                          SentenceSplitter,
                                          SentenceWindowNodeParser,
                                          HierarchicalNodeParser
                                          )
from llama_index.core import (VectorStoreIndex,
                              StorageContext,
                              PromptTemplate,
                              StorageContext,
                              SimpleDirectoryReader,
                              Document)
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline

from app.pdf_processing_utils import get_page_text, is_int
from app.utils import get_rag_prompt, get_retrieval_prompt
from app.text_cleaning_helpers import clean as advance_clean

from config import (OLLAMA_MODEL_NAME,
                    DEBUG,
                    EMBEDDING_MODEL_PATH,
                    QDRANT_URL,
                    QDRANT_API_KEY,
                    QDRANT_COLLECTION_NAME
                    )




def prepare_vector_store(store_collection_name,vector_db="qdrant"):
    if vector_db == "qdrant":
        client = QdrantClient(location=QDRANT_URL,api_key=QDRANT_API_KEY)
        async_client = AsyncQdrantClient(location=QDRANT_URL,api_key=QDRANT_API_KEY)
        vector_store = QdrantVectorStore(client=client,aclient=async_client,collection_name=store_collection_name)
    else:
        raise ValueError (f"Vector db : {vector_db} not supported. Pick 'qdrant")
    print(f'{"==="*10} Vectore Store created using vector db {vector_db}')
    return vector_store


def create_vector_collection(nodes, store_collection_name, vector_db="qdrant"):
    vector_store = prepare_vector_store(store_collection_name,vector_db)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes,storage_context=storage_context)


def load_vector_collection(store_collection_name, embed_model, vector_db="qdrant"):
    vector_store = prepare_vector_store(store_collection_name,vector_db=vector_db)
    return VectorStoreIndex.from_vector_store(embed_model=embed_model,vector_store=vector_store)


def save_simple_doc_store(document_nodes,persist_fn):

    doc_store = SimpleDocumentStore()
    doc_store.add_documents(document_nodes)
    
    storage_context = StorageContext.from_defaults(docstore=doc_store)
    storage_context.persist(os.path.join("data",persist_fn))


def get_doc_from_store(persist_fn):
    docstore =  SimpleDocumentStore.from_persist_dir(persist_dir=os.path.join("data",persist_fn))
    documents = list(docstore.docs.values())
    return documents

def get_storage_contex(persist_fn):
    return StorageContext.from_defaults(persist_dir=os.path.join("data", persist_fn))


def basic_clean(txt):
    txt = txt.replace('-\n','') # remove line hyphenated words
    txt = re.sub(r'(?<!\n)\n(?!\n|[A-Z0-9])', ' ', txt) # remove unnecessary line break by merge sentence which starts with lower case
    txt = '\n\n'.join([line for line in txt.split('\n') if not is_int(line)]) # remove line whihc only have number, most likely a page number
    return txt


def prepare_document(pdf_paths,method="simple"):

    cleaning_func = partial(advance_clean,
            extra_whitespace=True,
            broken_paragraphs=True,
            bullets=True,
            ascii=True,
            lowercase=False,
            citations=True,
            merge_split_words=True,
            )

    pattern = os.path.join(pdf_paths, "*.pdf")
    pdf_files = glob(pattern)

    if DEBUG:
        pdf_files = pdf_files[:3]

    try:
        documents = None
        if method == "simple":
            documents = SimpleDirectoryReader(input_files=pdf_files).load_data()  
            for doc in documents:
                doc.text = basic_clean(doc.text)
                doc.text = cleaning_func(doc.text)


        elif method == "manual_parsing":
            documents = []
            for pdf in pdf_files:
                fn = os.path.basename(pdf).split('.')[0]
                documents.extend([Document(text=cleaning_func(page["text"]), metadata=page["metadata"]) for page in get_page_text(pdf,fn)])

                if DEBUG:
                    print(f'Text extraction completed from PDF document at path {pdf}')
        else:
            raise ValueError (f"Invalid Method : {method} not supported. Pick one of 'simple' or 'manual_parsing'")
        return documents
    except Exception as e:
        print(f"An error occurred while creating Document from pdf files: {e}")


def get_node_parser(embed_model, parsing_method="semantic", **kwargs):
    if parsing_method == "semantic":
        # If the distance between two adjacent sentence groups exceeds the breakpoint threshold, it indicates a semantic shift and marks the start of a new node.
        node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    elif parsing_method == "simple":
        # Check for required kwargs and provide default values or raise error
        chunk_size = kwargs.get("chunk_size")
        chunk_overlap = kwargs.get("chunk_overlap")
        if chunk_size is None or chunk_overlap is None:
            raise ValueError("chunk_size and chunk_overlap must be provided for 'simple' parsing method")
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif parsing_method == "sentence_window":

        window_size = kwargs.get("window_size")
        window_metadata_key = kwargs.get("window_metadata_key")
        original_text_metadata_key = kwargs.get("original_text_metadata_key")
        if window_size is None or window_metadata_key is None or original_text_metadata_key is None:
            raise ValueError("window_size, window_metadata_key, and original_text_metadata_key  must be provided for 'sentence_window' parsing method")
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
        )
    elif parsing_method == "hierarchical":
        chunk_sizes = kwargs.get("chunk_sizes")
        if chunk_sizes is None or type(chunk_sizes) is not list:
            raise ValueError("list of chunk_sizes must be provided for  hierarchical parsing method")
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    else: 
        raise ValueError(f'Invalid Parsing Method: {parsing_method}, choose one of "semantic", "simple", "sentence_window", "hierarchical"')
    
    return node_parser


def ingest(documents,node_parser_type="semantic",**kwargs):
    if node_parser_type == "semantic":
        semantic_node_parser = get_node_parser(Settings.embed_model, parsing_method="semantic")
        transformation = [semantic_node_parser,Settings.embed_model]
    elif node_parser_type == "sentence_window":
        sent_window_node_parser = get_node_parser(Settings.embed_model,
                                                  parsing_method="sentence_window",
                                                  **kwargs
                                                  )
        transformation = [sent_window_node_parser,Settings.embed_model]
    elif node_parser_type == "hierarchical":
        hierarchical_node_parser = get_node_parser(Settings.embed_model, parsing_method="hierarchical",**kwargs)
        transformation = [hierarchical_node_parser,Settings.embed_model]
    else:
        raise ValueError()

    pipeline = IngestionPipeline(transformations=transformation,**kwargs)
    return pipeline.run(nodes=documents)


def get_simple_query_pipeline(retriever, retrieval_prompt_tmp, rag_prompt_template, verbose=True):

    # Function to concatenate retrieved nodes
    def concatenate_nodes(retrieved_nodes):
        return "\n\n".join([node.node.get_content() for node in retrieved_nodes])

    # Create a FunctionComponent for concatenation
    concat_component = FunctionComponent(concatenate_nodes)

    query_pipeline = QueryPipeline(
        verbose=verbose
        )
    query_pipeline.add_modules(
        {
            "input":InputComponent(),
            "retrieval_prompt_tmp":retrieval_prompt_tmp,
            "retriever":retriever,
            "concat_component": concat_component,
            "rag_prompt_template":rag_prompt_template,
            "llm":Settings.llm
        }
    )

    query_pipeline.add_link("input","retrieval_prompt_tmp",dest_key="topic")
    query_pipeline.add_link("retrieval_prompt_tmp","retriever")
    query_pipeline.add_link("retriever","concat_component")
    query_pipeline.add_link("input","prompt_tmpl2",dest_key="input")
    query_pipeline.add_link("concat_component","rag_prompt_template",dest_key="context_str")
    query_pipeline.add_link("rag_prompt_template","llm")
    return query_pipeline


def build_query_pipeline(rag_type,base_index,similarity_top_k=5,num_queries=3,**kwargs):
    
    # Patching get queries function to edit the query prompt based on embedding types
    def get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)

        # assume LLM proper put each query on a newline
        queries = response.text.split("\n")
        queries = [self.prompt_str1.format(input=q.strip()) for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
            return [QueryBundle(q) for q in queries[: self.num_queries - 1]]
    
    # Create a FunctionComponent for concatenation
    verbose = kwargs.get("verbose",True)

    if rag_type == "simple":
        retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)

        if num_queries > 1:
            # 3. Create QueryTransformationRetriever
            transform_retriever = QueryTransformationRetriever(
                retriever=base_retriever,
                transform_queries=True,
                num_queries=3,  # Will generate 2 additional queries
                use_original_query=True,
            )

        retrieval_prompt_template = PromptTemplate(get_retrieval_prompt())
        rag_prompt_template = PromptTemplate(get_rag_prompt())

        query_pipeline = get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)

    elif rag_type == "fusion_retrieval":
        
        doc_store_name = kwargs.get("doc_store_name")
        if doc_store_name is None:
            raise ValueError("doc_store_name is required parameter for fusion retrieval")
        doc_store = get_storage_contex(doc_store_name).docstore
        
        # retrieval based on embeddings
        vector_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
        
        # retrieval based on keywords
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=doc_store, similarity_top_k=similarity_top_k
        )

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=4,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )

        retriever.prompt_str1 = "Represent this sentence for searching relevant passages: {input}"
        retriever._get_queries = MethodType(get_queries, retriever)

        retrieval_prompt_template = PromptTemplate(get_retrieval_prompt(None))
        rag_prompt_template = PromptTemplate(get_rag_prompt())
        query_pipeline = get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)
    
    elif rag_type == "small_to_big":
        
        doc_store_name = kwargs.get("doc_store_name")

        print(f'Ensure Node store in doc_store {doc_store_name} has parent-child relation otherwise this retrieval will be same as Simple RAG')
        if doc_store_name is None:
            raise ValueError("doc_store_name is required parameter for fusion retrieval")
        
        storage_context = get_storage_contex(doc_store_name)
        base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=verbose)

        retrieval_prompt_template = PromptTemplate(get_retrieval_prompt(None))
        rag_prompt_template = PromptTemplate(get_rag_prompt())
        query_pipeline = get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)
    