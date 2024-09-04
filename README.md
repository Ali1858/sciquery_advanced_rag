# SciQuery : Advanced RAG System using LlamaIndex


TODO: 
- [ ] Test Docker script.
- [ ] Improve PDF parsing.
- [ ] Add Support for more retrievers such as `Recursive Retrieval`, `Automated Metadata Extraction` and `Enhanced Query Generation`

This repository demonstrates the use of the llama index library for Advance Retrieval-Augmented Generation (RAG), specifically designed for scientific literature Q&A. It showcases a multiple RAG system, building using several chunking and retrieval methods. In addition to that it also uses Qdrant vector database to persist embeddings. This approach help in understanding how different RAG technique works. The system also provides an API for updating, deleting, and quering documents in the index.

## Steps to Implement SciQuery:

To get started, you need to download scientific literatures and save it in the `$PDF_DATA_DIR` folder for initial indexing. The application will creates an index of all documents using several chunking methods and persist their embeddings along with metadata on Qdrant vector DB.

### Step 1: PDF Processing:
The first step involves processing PDF documents and converting them into text. We provide two options, one, default `llama_index's` pdf parsing and second using the `PyMuPDF` library to extract text from pdfs.

We use regex to identify the starting point of the `Reference` section, which is useful for cleaning bibliography from each paper.

### Step 2: Chunking:
The system provides several chunking methods to process and divide text into manageable segments or "nodes". Suitable method can be choosen after emperical testing: to different use cases and requirements:

  1. Semantic Chunking (semantic):
    This method splits text based on semantic shifts. It uses an embedding model to detect changes in meaning, creating a new chunk whenever a significant semantic difference is identified. This is useful for tasks where preserving the coherence of meaning is important.

  2. Simple Chunking:
    This method divides text into fixed-size chunks, with a specified overlap between consecutive chunks. It is straightforward and useful for scenarios where equal-sized segments are needed, regardless of content. 

  3. Sentence Window Chunking (sentence_window):
    This method uses a sliding window to generate chunks, moving over the text with a specified window size. Each window captures a group of sentences, allowing for overlapping segments that provide context continuity. 
  
  4. Hierarchical Chunking (hierarchical):
    This method creates a hierarchical structure of chunks, using different sizes for different levels. It enables a layered representation of text, which is beneficial for complex tasks requiring multi-level analysis or summary. It uses predefined chunk sizes for different hierarchy levels (e.g., 2048, 512, 128).

### Step 3: Embedding:

For effective semantic search, it is crucial to obtain vector embeddings of text passages. We generate these embeddings using the [Sentence Transformer](https://pypi.org/project/sentence-transformers/) library and [MixedBread Embedding](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).

These embeddings, along with the original text passages and their metadata, are saved on Qdrant vector store.

### Step 4: Retrieval Techniques:
we provide variety of Retrieval-Augmented Generation (RAG) techniques to enhance the effectiveness of text retrieval and generation. These techniques are designed to improve the quality and relevance of generated responses by utilizing structured retrieval processes. Below is a brief explanation of each supported RAG technique:

  1. RAG with Small-to-Big (Hierarchical Indexing): 
    This technique utilizes a hierarchical structure to organize and index documents. By segmenting documents into different levels of granularity, it enables efficient and precise retrieval of relevant information based on the query context. This hierarchical approach enhances the ability to handle large and complex documents by focusing on the most relevant sections first and then expanding to more detailed levels as needed.
    
  2. RAG with Smal-to-Big Retrieval (Sentence Window Indexing): 
    This method involves indexing documents at a very fine-grained level, such as individual sentences or small segments of text. By using a sliding window approach, it allows the system to quickly retrieve the most relevant sentences that match the query and then expanding to its window of contex. This technique is particularly useful for extracting precise pieces of information from documents, improving the specificity and accuracy of the responses generated.
  
  3. Fusion Retrieval: 
    Fusion Retrieval combines multiple retrieval strategies to leverage the strengths of each. It aggregates results from various indexing and retrieval techniques, merging the outcomes to form a comprehensive response. This approach ensures that diverse sources of information are considered, leading to more robust and well-rounded generated content. We use fusion of semantic retrieval and BM25 retrieval in this task.

  4. Simple RAG:
    Simple RAG involves a straightforward approach to retrieval-augmented generation, where a single retrieval method is used to find relevant documents or text segments, which are then fed into the generation process. While less complex than other techniques, Simple RAG can still effectively enhance the relevance of responses, especially when dealing with well-defined and narrower queries.

  [TODO] Implementation pending.

  5. Contextual Compression:

  6. Adaptive Retrieval:

  7. Sophisticated Controllable Agent:
  
  8. Recursive Retrieval:
  
  9. Context Enrichment using QA and Summary:
  
  10. DocumentSummaryIndex Retrieval:

## REST API Endpoints

The SciQuery system exposes several REST API endpoints to manage and query the document index. Below is a description of each endpoint, along with example `curl` commands for interacting with them.

### 1. **Manage Index**

This resource handles operations related to managing the document index, including retrieving, adding, and deleting documents.
- **GET `/api/documents`**

  Fetches a list of document names exits in Index. These file name can be use to delete a document from INDEX.

  **Example `curl` Command:**
  ```bash
  curl -X GET "http://127.0.0.1:5000/api/documents"
  ```

  **Sample Response**
  ```json
  {
  "file_names": [
    "conneau2017_word_translation_without_parallel_data_muse_csls.pdf",
    "martins2016_sparsemax.pdf",
    "lample2019-cross-lingual-language-model-pretraining-Paper.pdf",
    ]
  }
  ```

- **DELETE `/api/documents/<string:file_name>`**

  Deletes a document from the index by its file_name. Replace `<string:file_name>` with the actual file name of the document you want to delete.

  **Example `curl` Command:**
  ```bash
  curl -X DELETE "http://127.0.0.1:5000/api/documents/martins2016_sparsemax.pdf"
  ```

  **Sample Response**
  ```json
  {
    "message": "Document with file name: martins2016_sparsemax.pdf is successfully deleted from the index"
  }

  ```
  **You can see File with UUID "martins2016_sparsemax.pdf" is deleted from Index**
  ```json
  {
  "file_names": [
    "conneau2017_word_translation_without_parallel_data_muse_csls.pdf",
    "lample2019-cross-lingual-language-model-pretraining-Paper.pdf"
    ]
  }
  ```

- **POST `/api/documents`**

  Adds a new PDF document to the index. The PDF file should be included in the form-data under the key `file`.

  **Example `curl` Command:**
  ```bash
  curl -X POST "http://127.0.0.1:5000/api/documents" \
  -F "file=@/path/to/martins2016_sparsemax.pdf"
  ```

  **Sample Response**
  ```json
  {
    "message": "Document with file name: martins2016_sparsemax.pdf is successfully added to the index"
  }
  ```

  **You can see "martins2016_sparsemax.pdf" is added back to the Index**
  ```json
  {
  "file_names": [
    "conneau2017_word_translation_without_parallel_data_muse_csls.pdf",
    "martins2016_sparsemax.pdf",
    "lample2019-cross-lingual-language-model-pretraining-Paper.pdf"
    ]
  },
  ```


### 2. **Query Index**

This resource processes queries to retrieve relevant information from the indexed documents.

- **POST `/api/query`**

  Sends a query to retrieve relevant passages and generate an answer. The query should be provided in the JSON body under the key `query`.

  **Example `curl` Command:**
  ```bash
  curl -X POST "http://127.0.0.1:5000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain what is ULMFIT?"}'
  ```

  **Sample Response with [TRUNCATED] passage**
  ```json
  {
    "relevant_nodes": [
        {
            "text": "We ne-tune the classier for 50epochs and train all methods but ULMFiT with early stopping. Low-shot learning One of the main benets of transfer learning is being able to train a .... However, even for large datasets, pretraining improves performance.",
            "node_id": "d0d11687-c32f-4e26-a636-9f628ef2880a",
            "page_num": "7",
            "file_name": "ruder2019_ulmfit.pdf"
        },
        {
            "text": "Figure 4: Validation error rate curves for netuning the classier with ULMFiT and Full on IMDb, TREC-6, and AG (top to bottom). The error then increases as the model starts to overt ... will catalyze new developments in transfer learning for NLP. Acknowledgments We thank the anonymous reviewers for their valuable feedback. ",
            "node_id": "5636843b-954f-497e-af75-e8fecc45ba02",
            "page_num": "9",
            "file_name": "ruder2019_ulmfit.pdf"
        },
        {
            "text": "13/02/2018ulmt_pretraining.html 1/1dollarThegoldorEmbeddinglayerLayer1Layer2Layer3Softmaxlayer gold(a) LM pre-training 13/02/2018ulmt_lm_ne-tuning.html ... ne-tuned on target task data using discriminative ne-tuning ( Discr ) and slanted triangular learning rates (STLR) to learn task-specic features. ",
            "node_id": "436fc1e3-5810-4520-b269-2f10d998eab5",
            "page_num": "3",
            "file_name": "ruder2019_ulmfit.pdf"
        },
        {
            "text": "329fer learning approaches on six widely studied text classication tasks. On IMDb, with 100labeled examples, ULMFiT matches the performance of training from scratch with 10andgiven ... of all CNN units above that pixel. In analogy, a hypercolumn for a word or sentence in NLP is the concatenation of embeddings at different layers in a pretrained model.et al. (2017), and McCann et al. ",
            "node_id": "98daa1fc-67b0-4c4a-9d48-e2b7c23bdb74",
            "page_num": "2",
            "file_name": "ruder2019_ulmfit.pdf"
        },
        {
            "text": "(2018) require engineered custom architectures, while we show state-of-the-art performance with the same basic architecture across a range of tasks. In CV ,... entailment (Conneau et al., 2017), it provides data in near-unlimited quantities for most domains and languages. Additionally, a pretrained LM can be easily adapted to the idiosyncrasies of a target",
            "node_id": "d075beb8-0825-4e7e-a872-a1a30851d7e2",
            "page_num": "2",
            "file_name": "ruder2019_ulmfit.pdf"
        }
    ],
    "query": "Explain what is ULMFIT?",
    "answer": "ULMFiT (Universal Language Model Fine-tuning) is a transfer learning method for Natural Language Processing (NLP) tasks. It consists of three stages: \n\n1. Pretraining: A language model is trained on a general-domain corpus to capture general features of the language.\n2. LM pre-training: The full language model is ne-tuned on target task data using discriminative tuning and slanted triangular learning rates to learn task-specific features.\n3. Classier ne-tuning: The classier layer is fine-tuned on the target task data.\n\nULMFiT allows for extremely sample-efficient transfer learning and achieves state-of-the-art results on six representative text classification tasks, including IMDb, TREC-6, and AG News."
  }
  ```

## Installation and Setup

To get started with SciQuery, follow these steps to set up your Python environment, install the required dependencies, and start the Flask application.

1. **Create a Python Virtual Environment**

   First, create a virtual environment to manage your project's dependencies. Run the following command:

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**

   Activate the virtual environment. The command depends on your operating system:

   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install Required Dependencies**

   With the virtual environment activated, install the necessary packages using `pip`. Make sure to also install [MLX](https://pypi.org/project/mlx-llm/) library for Apple Silicon. This library is useful to load quantized model on Apple Silicon.:

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask Application**

   Finally, start the Flask application using the following command:

   ```bash
   flask run
   ```

Make sure to set any necessary environment variables as specified in the `config.py` file before running the application.

Replace `http://localhost:5000` with the actual base URL of your SciQuery API server. For the `POST` and `DELETE` requests, make sure to use the appropriate file paths and UUIDs as needed.

Thank you for exploring the SciQuery project. If you have any questions or contributions, please feel free to reach out.
