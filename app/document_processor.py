import os
from glob import glob
import re
from typing import List
from functools import partial

from llama_index.core import SimpleDirectoryReader, Document

from app.pdf_processing_utils import get_page_text, is_int
from app.text_cleaning_helpers import clean as advanced_clean

from config import DEBUG

class DocumentProcessor:
    """Handles document preparation and cleaning."""
    
    def __init__(self, debug: bool = DEBUG):
        self.debug = debug
        self.cleaning_func = partial(advanced_clean,
                                     extra_whitespace=True,
                                     broken_paragraphs=True,
                                     bullets=True,
                                     ascii=True,
                                     lowercase=False,
                                     citations=True,
                                     merge_split_words=True)

    def basic_clean(self, txt: str) -> str:
        """Applies basic cleaning operations to the provided text."""
        txt = txt.replace('-\n', '')  # remove line hyphenated words
        txt = re.sub(r'(?<!\n)\n(?!\n|[A-Z0-9])', ' ', txt)  # remove unnecessary line break
        txt = '\n\n'.join([line for line in txt.split('\n') if not is_int(line)])  # remove lines that only have numbers
        return txt
        

    def extract_reference_section_text(self, page_text):
        found_references = False
        pattern = r'(?:References|Reference|Reference:|References:|Bibliography|Bibliographical References)\s*\n'
        
        if re.search(pattern, page_text, flags=re.IGNORECASE):
            found_references = True
            modified_text = re.sub(pattern + r'.*', '', page_text, flags=re.IGNORECASE | re.DOTALL)
        else:
            modified_text = page_text  # No change if no reference section is found
        
        return modified_text, found_references


    def prepare_documents(self, pdf_dir: str, parse_doc_again=True, method: str = "simple") -> List[Document]:
        """Prepares documents from PDF files located at the specified path."""
        print(pdf_dir)
        pattern = os.path.join(pdf_dir, "*.pdf")
        pdf_files = glob(pattern)
        
        if self.debug:
            pdf_files = pdf_files[:2]
            print(pdf_files)
        
        documents = []

        if parse_doc_again:
            if method == "simple":
                documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
                for doc in documents:
                    doc.text = self.basic_clean(doc.text)
                    # doc.text, found_references = self.extract_reference_section_text(doc.text)
                    doc.text = self.cleaning_func(doc.text)
            elif method == "manual_parsing":
                for pdf in pdf_files:
                    fn = os.path.basename(pdf).split('.')[0]
                    documents.extend([Document(text=self.cleaning_func(page["text"]), metadata=page["metadata"]) 
                                    for page in get_page_text(pdf, fn)])
                    if self.debug:
                        print(f'Text extraction completed from PDF document at path {pdf}')
            else:
                raise ValueError(f"Invalid Method: {method} not supported. Pick 'simple' or 'manual_parsing'")

            print(f'Found {len(documents)} total number of pages from the {len(pdf_files)} pdf files')
        else:
            print(f'Not parsing pdf document. Its already in doc store')

        return documents
    

    def prepare_single_document(self, pdf_file: str, method: str = "simple") -> List[Document]:
        """Prepares a single document from a PDF file located at the specified path."""
        if not os.path.isfile(pdf_file) or not pdf_file.endswith('.pdf'):
            raise ValueError(f"The file {pdf_file} is not a valid PDF file")

        documents = []
        if method == "simple":
            documents = SimpleDirectoryReader(input_files=[pdf_file]).load_data()
            for doc in documents:
                doc.text = self.basic_clean(doc.text)
                doc.text = self.cleaning_func(doc.text)
        elif method == "manual_parsing":
            fn = os.path.basename(pdf_file).split('.')[0]
            documents.extend([Document(text=self.cleaning_func(page["text"]), metadata=page["metadata"]) 
                            for page in get_page_text(pdf_file, fn)])
            if self.debug:
                print(f'Text extraction completed from PDF document at path {pdf_file}')
        else:
            raise ValueError(f"Invalid Method: {method} not supported. Pick 'simple' or 'manual_parsing'")
        print(f'Found {len(documents)} total number of pages from the document {pdf_file}')
