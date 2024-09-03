from llama_index.core.node_parser import (SemanticSplitterNodeParser, SentenceSplitter,
                                          SentenceWindowNodeParser, HierarchicalNodeParser)

class NodeParserFactory:
    """Factory for creating different types of node parsers."""
    
    @staticmethod
    def get_node_parser(embed_model, parsing_method: str = "semantic", **kwargs):
        """Returns a node parser based on the specified parsing method."""
        if parsing_method == "semantic":
            return SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
        elif parsing_method == "simple":
            chunk_size = kwargs.get("chunk_size")
            chunk_overlap = kwargs.get("chunk_overlap")
            if chunk_size is None or chunk_overlap is None:
                raise ValueError("chunk_size and chunk_overlap must be provided for 'simple' parsing method")
            return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif parsing_method == "sentence_window":
            window_size = kwargs.get("window_size")
            window_metadata_key = kwargs.get("window_metadata_key")
            original_text_metadata_key = kwargs.get("original_text_metadata_key")
            if window_size is None or window_metadata_key is None or original_text_metadata_key is None:
                raise ValueError("window_size, window_metadata_key, and original_text_metadata_key must be provided for 'sentence_window' parsing method")
            return SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key=window_metadata_key,
                original_text_metadata_key=original_text_metadata_key,
            )
        elif parsing_method == "hierarchical":
            chunk_sizes = kwargs.get("chunk_sizes")
            if chunk_sizes is None or type(chunk_sizes) is not list:
                raise ValueError("list of chunk_sizes must be provided for hierarchical parsing method")
            return HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        else:
            raise ValueError(f'Invalid Parsing Method: {parsing_method}, choose one of "semantic", "simple", "sentence_window", "hierarchical"')

