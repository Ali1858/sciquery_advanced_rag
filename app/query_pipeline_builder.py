
from types import MethodType 
import nest_asyncio

from llama_index.core.query_pipeline import InputComponent, FunctionComponent, QueryPipeline
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.schema import QueryBundle

from app.utils import get_rag_prompt, get_retrieval_prompt
from app.vector_store_manager import VectorStoreManager

class QueryPipelineBuilder:
    """Builds different types of query pipelines."""

    def __init__(self, Settings, node_parsing_method=None, num_queries=3, verbose = False) -> None:
        
        self.num_queries = num_queries
        self.Settings = Settings
        self.llm = Settings.llm
        self._verbose = verbose
        self.node_parsing_method = node_parsing_method
        self.QUERY_GEN_PROMPT = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
            )
    

    def get_queries(self, *args):

        if len(args) == 2:
            reference_patched_class, original_query = args
        else:
            original_query = args

        prompt_str = self.QUERY_GEN_PROMPT.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self.llm.complete(prompt_str)

        # assume LLM proper put each query on a newline
        queries = response.text.split("\n")
        queries = [self.query_gen_retrieval_prompt.format(topic=q.strip()) for q in queries if q.strip() and q.strip()[0].isdigit()] 
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]
    

    def get_simple_query_pipeline(self, retriever, retrieval_prompt_tmp, rag_prompt_template, verbose: bool = True):
        """Creates and returns a simple query pipeline."""
        node_parsing_method = self.node_parsing_method
        if node_parsing_method == "sentence_window":
            print('Adding function to fetch window for sentence window small-to-big retrieval')

        def concatenate_nodes(retrieved_nodes):
            if node_parsing_method == "sentence_window":
                return "\n\n".join([node.node.metadata["window"] for node in retrieved_nodes])
            else:
                return "\n\n".join([node.node.get_content() for node in retrieved_nodes])

        
        concat_component = FunctionComponent(concatenate_nodes)
        query_pipeline = QueryPipeline(verbose=verbose)
        query_pipeline.add_modules({
            "input": InputComponent(),
            "retrieval_prompt_tmp": retrieval_prompt_tmp,
            "retriever": retriever,
            "concat_component": concat_component,
            "rag_prompt_template": rag_prompt_template,
            "llm": self.Settings.llm,
        })

        query_pipeline.add_link("input", "retrieval_prompt_tmp", dest_key="topic")
        query_pipeline.add_link("retrieval_prompt_tmp", "retriever")
        query_pipeline.add_link("retriever", "concat_component")
        query_pipeline.add_link("input", "rag_prompt_template", dest_key="input")
        query_pipeline.add_link("concat_component", "rag_prompt_template", dest_key="context_str")
        query_pipeline.add_link("rag_prompt_template", "llm")
        return query_pipeline
    

    def build_query_pipeline(self, rag_type: str, base_index, similarity_top_k: int = 5, num_queries: int = 3, **kwargs):
        """Builds a query pipeline based on the specified RAG type."""
        verbose = kwargs.get("verbose", False)

        if rag_type == "simple":
            retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
            retrieval_prompt_template = PromptTemplate(get_retrieval_prompt())
            rag_prompt_template = PromptTemplate(get_rag_prompt())
            return self.get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)

        elif rag_type == "fusion_retrieval":
            # apply nested async to run in a notebook
            nest_asyncio.apply()
            doc_store = kwargs.get("doc_store")
            if doc_store is None:
                raise ValueError("doc_store is a required parameter for fusion retrieval")
           
            vector_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
            bm25_retriever = BM25Retriever.from_defaults(docstore=doc_store, similarity_top_k=similarity_top_k)
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=3,
                mode="reciprocal_rerank",
                use_async=True,
                verbose=verbose,
                retriever_weights = [0.75,0.25]
            )

            retriever._get_queries = MethodType(self.get_queries, retriever)
            retrieval_prompt_template = PromptTemplate(get_retrieval_prompt(None))
            self.query_gen_retrieval_prompt = get_retrieval_prompt()
            rag_prompt_template = PromptTemplate(get_rag_prompt())
            return self.get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)

        elif rag_type == "small_to_big":
            storage_context = kwargs.get("storage_context")
            if storage_context is None:
                raise ValueError("storage_context is a required parameter for hierarchical retrieval")
            
            base_retriever = base_index.as_retriever(similarity_top_k=similarity_top_k)
            retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=verbose)

            retrieval_prompt_template = PromptTemplate(get_retrieval_prompt(None))
            rag_prompt_template = PromptTemplate(get_rag_prompt())
            return self.get_simple_query_pipeline(retriever, retrieval_prompt_template, rag_prompt_template, verbose=verbose)
        else:
            raise ValueError(f"RAG Type {rag_type} not supported. Pick one from 'simple', 'fusion_retrieval' and 'small_to_big'.")