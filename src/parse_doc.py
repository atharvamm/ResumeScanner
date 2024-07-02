import os
from typing import List, Dict
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pdfplumber

def text_splitter(data: List[Document]) -> List[Document]:
    """
    Splits the content of the documents into chunks of specified size and overlap.

    Args:
        data (List[Document]): List of documents to be split.

    Returns:
        List[Document]: List of split documents with updated metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    
    for i, text in enumerate(texts):
        text.metadata["doc_seq"] = i + 1
    
    return texts

def parse_pdf(doc_path: str) -> List[Document]:
    """
    Parses a PDF document and extracts its text content.

    Args:
        doc_path (str): Path to the PDF document.

    Returns:
        List[Document]: List containing a single Document object with the extracted text.
    """
    pdf = pdfplumber.open(doc_path)
    pages = [page.extract_text() for page in pdf.pages]
    string = "\n".join(pages)
    data = [Document(page_content=string, metadata={"source": doc_path})]
    
    return data

def parse_docx(doc_path: str) -> List[Document]:
    """
    Parses a DOCX document and extracts its text content.

    Args:
        doc_path (str): Path to the DOCX document.

    Returns:
        List[Document]: List containing a single Document object with the extracted text.
    """
    loader = Docx2txtLoader(doc_path)
    data = loader.load()
    
    return data

def parse_doc(doc_path: str) -> List[Document]:
    """
    Parses a DOC document and extracts its text content.

    Args:
        doc_path (str): Path to the DOC document.

    Returns:
        List[Document]: List containing a single Document object with the extracted text.
    """
    loader = UnstructuredWordDocumentLoader(doc_path, mode="single")
    data = loader.load()
    
    return data

def parse_dir(path: str) -> List[Document]:
    """
    Parses all the documents in a specified directory and extracts their text content.

    Args:
        path (str): Path to the directory containing the documents.

    Returns:
        List[Document]: List of Document objects with the extracted text from each document.
    """
    # Mapping of file extensions to their respective parsing functions
    doc_funcs: Dict[str, callable] = {
        "doc": parse_doc,
        "docx": parse_docx,
        "pdf": parse_pdf
    }

    docs_data: List[Document] = []
    doc_list = os.listdir(path)
    
    for doc_name in doc_list:
        if "DS_Store" in doc_name or doc_name.startswith("~"):
            continue
        
        doc_path = os.path.join(path, doc_name)

        if doc_name.startswith("__"):
            continue

        ext = doc_name.split(".")[-1]
        
        if ext in doc_funcs:
            data = doc_funcs[ext](doc_path)
            docs_data.extend(data)
    
    return docs_data
