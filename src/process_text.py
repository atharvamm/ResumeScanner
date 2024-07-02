import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.schema.document import Document  

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters except certain allowed symbols.
    
    Args:
    - text (str): The input text to be cleaned.
    
    Returns:
    - str: Cleaned text with special characters removed.
    """
    text = re.sub(r'[^\w\s+-.,\'\/]', '', text)  # Removes special characters except '+', '-', '.', ',', ''', and '/'
    return text

def process_text(text: str) -> list:
    """
    Processes the input text by converting it to lowercase, cleaning special characters,
    and splitting into tokens.
    
    Args:
    - text (str): The input text to be processed.
    
    Returns:
    - list: List of tokens (words) extracted from the processed text.
    """
    text = text.lower()  # Convert text to lowercase
    text = clean_text(text)  # Clean text using the clean_text function
    tokens = text.split(" ")  # Split text into tokens (words) based on spaces
    return tokens  # Return list of tokens

def get_doc_frm_st(string: str = "", source: str = "") -> list:
    """
    Constructs a list containing a Document object with provided page content and metadata.
    
    Args:
    - string (str): Page content to be stored in the Document object.
    - source (str): Source information to be stored in the Document metadata.
    
    Returns:
    - list: List containing a single Document object initialized with the provided content and metadata.
    """
    return [Document(page_content=string, metadata={"source": source})]

