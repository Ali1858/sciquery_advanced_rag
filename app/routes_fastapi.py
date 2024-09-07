import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from werkzeug.utils import secure_filename
from config import PDF_DATA_DIR
from app.rag_pipeline_manager import RAGPipelineManager

app = FastAPI()


# Model to handle queries
class QueryRequest(BaseModel):
    query: str

# Function to get RAG Manager
def get_rag_manager(request: Request) -> RAGPipelineManager:
    return request.app.rag_manager


@app.get("/api/documents", response_model=List[str])
async def get_documents(request: Request):
    "Get request to fetch list of file names of all PDF documents in our Index."
    try:
        rag_manager = get_rag_manager(request)
        print(f'Total nodes in docstore {rag_manager.doc_store_name}: {len(rag_manager.doc_store.docs)}')
        files = list(set([node.metadata["file_name"] for _, node in rag_manager.doc_store.docs.items()]))
        return files
    except Exception as e:
        print(f'Error occurred while fetching document names in the Index: {e}')
        raise HTTPException(status_code=500, detail="Error while fetching PDF documents in the Index")


@app.post("/api/documents")
async def add_document(request: Request, file: UploadFile = File(...)):
    "Post request to add a new document to the Index."
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        filename = secure_filename(file.filename)
        rag_manager = get_rag_manager(request)

        existing_files = list(set([node.metadata["file_name"] for _, node in rag_manager.doc_store.docs.items()]))
        if filename in existing_files:
            raise HTTPException(status_code=400, detail=f"Document with file name {filename} already exists in the Index")

        filepath = os.path.join(PDF_DATA_DIR, filename)
        
        # Save the file
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        # Update the index
        rag_manager.update_for_one_doc(filepath)

        return {"message": f"Document {filename} successfully added to the index"}
    except Exception as e:
        print(f'Error occurred while adding document: {e}')
        raise HTTPException(status_code=500, detail="Error while adding the document")


@app.delete("/api/documents/{file_name}")
async def delete_document(request: Request, file_name: str):
    "Delete a document from the Index by file name."
    try:
        rag_manager = get_rag_manager(request)
        if rag_manager.delete_doc_from_index(file_name):
            return {"message": f"Document {file_name} successfully deleted from the index"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {file_name} doesn't exist in the Index")
    except Exception as e:
        print(f'Error occurred while deleting document {file_name}: {e}')
        raise HTTPException(status_code=500, detail="Error while deleting the document")


@app.post("/api/query")
async def query_index(request: Request, query_data: QueryRequest):
    "Post request to query the Index."
    try:
        rag_manager = get_rag_manager(request)
        query = query_data.query
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        _, response = rag_manager.run_query_pipeline(query)

        return JSONResponse(content=response)
    except Exception as e:
        print(f'Error occurred while querying: {e}')
        raise HTTPException(status_code=500, detail=f"Error generating answer for query: {query_data.query}")


