import os
from flask import jsonify, request
from flask_restful import Resource, current_app
from werkzeug.utils import secure_filename
from flask import Flask
from flask_restful import Api

from app.rag_pipeline_manager import RAGPipelineManager
from config import RAG_DOCUMENT_PARSING_METHOD, RAG_NODE_PARSING_METHOD, RAG_RETRIEVAL_TYPE, PDF_DATA_DIR


class ManageIndex(Resource):
    def get(self):
        "Get request to fetch dictionary of uuid and filename of all PDF document in our Index."
        try:
            rag_manager = current_app.config.get("RAG_MANAGER")
            print(f'Total node save in docstore {rag_manager.doc_store_name} is {len(rag_manager.doc_store.docs)}')
            files = list(set([node.metadata["file_name"] for id, node in rag_manager.doc_store.docs.items()]))
            return jsonify({"file_names": files})
        except Exception as e:
            print(f'Error occur while getting uuid and document mapping:{e}.')
            return {"message":"Error while getting uuid and PDF document mapping"}, 500


    def post(self):
        "Post request to add new document in the Index."
        try:
            if 'file' not in request.files:
                return {'message': 'No file part in the request'}, 400
            file = request.files['file']
            if file.filename == '':
                return {'message': 'No file selected for uploading'}, 400
            if file and file.filename.lower().endswith('.pdf'):
                rag_manager = current_app.config.get("RAG_MANAGER")

                # Save PDF on disk
                filename = secure_filename(file.filename)

                if filename in list(set([node.metadata["file_name"] for id, node in rag_manager.doc_store.docs.items()])):
                    return {"message": f"Document with file name {filename} already exist in the Index."}, 400
                
                filepath = os.path.join(PDF_DATA_DIR, filename)
                file.save(filepath)
                
                rag_manager.update_for_one_doc(filepath)

                return {'message': f"Document with file name: {filename} is successfully added to the index"}, 201
            return {'message': 'Allowed file type is PDF'}, 400
        except Exception as e:
            print(f'Error occur while adding new document in the Index:{e}.')
            return {"message":"Error while adding new document in the Index."}, 500


    def delete(self, file_name):
        "Delete request to delete a document from the Index by its uuid"
        try:
            rag_manager = current_app.config.get("RAG_MANAGER")
            
            if rag_manager.delete_doc_from_index(file_name):
                return {"message": f"Document with file name: {file_name} is successfully deleted from the index"}, 201
            else:
                return {"message": f"Document with file name: {file_name} doesn't exist in index. Please use correct file name"}, 404
        except Exception as e:
            return {'message': f'Error occur while deleting the document with file name {file_name} in Index'}, 500


class QueryIndex(Resource):
    def post(self):
        try:
            query = request.json.get('query', '')
            if not query:
                return {'message': 'No query provided'}, 400
            
            rag_manager = current_app.config.get("RAG_MANAGER")
            _ , response = rag_manager.run_query_pipeline(query)

            return response, 200
        except Exception as e:
            print(e)
            return {'message': f'Error occur while generating a answer for query: {query}.'}, 500

