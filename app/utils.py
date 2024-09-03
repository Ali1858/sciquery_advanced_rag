from typing import List 

def get_retrieval_prompt(embedding_type="mix_bread"):
    if embedding_type == "mix_bread":
        return "Represent this sentence for searching relevant passages: {topic}"
    else:
        return "{topic}"


def get_rag_prompt() -> str:
    prompt = """
    <|begin_of_text|>
    <|start_header_id|>
        system
    <|end_header_id|>
        You are an AI assistant tasked with answering questions based on provided context.
        Your goal is to provide accurate, relevant, and concise responses using only the information given in the context.
        If the context doesn't contain enough information to answer the question fully, state that clearly. 
        Do not make up or infer information beyond what's explicitly stated in the context.

        [INSTRUCTIONS]
        1. Carefully read the provided context and query.
        2. Analyze the information in the context that is relevant to the query.
        3. Formulate a clear and concise answer based solely on the given context.
        4. If the context doesn't provide sufficient information to answer the query, state this explicitly.
        5. Do not include any information that is not present in the given context.
    <|eot_id|>
    <|start_header_id|>
        user
    <|end_header_id|>
    Answer the user question based on the context provided below

    [CONTEXT]:
    {context_str}
    
    [QUERY]
    {input}
    
    <|eot_id|>
    <|start_header_id|>
        assistant
    <|end_header_id|>
    """
    return prompt.strip()

