from src.parse_doc import text_splitter 
import numpy as np 
from src.process_text import get_doc_frm_st  
import os 
import re 
import src.util as util 
import src.string_util as string_util 
from langchain_community.embeddings import HuggingFaceEmbeddings 

def get_embedding(doc, embedding=None):
    """
    Function to get embeddings for a list of documents using HuggingFace models.

    Args:
    - doc (list): List of documents to embed.
    - embedding (HuggingFaceEmbeddings, optional): Embedding model instance. If None, default model is used.

    Returns:
    - np.ndarray: Array of embeddings for the input documents.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism
    
    # Initialize embedding model if not provided
    if embedding is None:
        try:
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except:
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", force_download=True)
    
    # Ensure doc is a list
    if not (isinstance(doc, list) or isinstance(doc, tuple)):
        doc = list(doc)
    
    # Return embeddings for the input documents
    return embedding.embed_documents(doc)


def normalize_vectors(vectors):
    """
    Function to normalize vectors.

    Args:
    - vectors (np.ndarray): Array of vectors to normalize.

    Returns:
    - np.ndarray: Normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def query_text_cosine_score(query_docs, text_docs, threshold=0.0):
    """
    Function to compute cosine similarity scores between query and text documents.

    Args:
    - query_docs (list): List of query documents.
    - text_docs (list): List of text documents.
    - threshold (float, optional): Similarity threshold for considering a match.

    Returns:
    - float: Cosine similarity score between query and text documents.
    """
    query_embeddings = np.array(get_embedding([query_doc.page_content for query_doc in query_docs]))
    text_embeddings = np.array(get_embedding([text_doc.page_content for text_doc in text_docs]))
    normalized_text_vecs = normalize_vectors(text_embeddings)

    total_score, query_length = 0, 0
    for query in query_embeddings:
        normalized_query = normalize_vectors(query.reshape(1, -1))

        scores = normalized_query @ normalized_text_vecs.T
        query_length += len(query)
        total_score += np.max(scores) * len(query)
    
    return round(((total_score / query_length + 1) / 2) * 100, 5)


def rag_novectorstore_skills(query, text, threshold=0.0):
    """
    Function to find relevant text documents based on query using cosine similarity.

    Args:
    - query (str): Query text.
    - text (str): Text to search for matches.
    - threshold (float, optional): Similarity threshold for considering a match.

    Returns:
    - str: Merged overlapping strings of relevant documents.
    """
    added_docs = set()
    ans = []

    # Split and process query and text documents
    query_docs = text_splitter(get_doc_frm_st(query))
    text_docs = text_splitter(get_doc_frm_st(text))

    query_embeddings = np.array(get_embedding([query_doc.page_content for query_doc in query_docs]))
    text_embeddings = np.array(get_embedding([text_doc.page_content for text_doc in text_docs]))

    normalized_doc_vecs = normalize_vectors(text_embeddings)

    for ind, query in enumerate(query_embeddings):
        normalized_query = normalize_vectors(query.reshape(1, -1))
        scores = normalized_query @ normalized_doc_vecs.T
        
        for score_ind, score in enumerate(scores[0]):
            if score > threshold:
                added_docs.add(score_ind)
    
    for index in sorted(added_docs):
        ans.append(text_docs[index].page_content)
    
    return string_util.merge_overlapping_strings(ans)


def get_jd_domain_reqs_prompt(data):
    """
    Function to extract technical requirements and domain from a job description using Gemini.

    Args:
    - data (str): Job description text.

    Returns:
    - tuple: Extracted requirements and domain as strings.
    """
    # Constructing prompt for Gemini API call
    text = '''
    Assume you are a senior IT recruiter with 20+ years of experience. Your task is to extract the technical requirements from the provided job description. The requirements noted should be as detailed and specific as, or better than, those in the job description. Additionally, identify the domain of the organization based on the job description (e.g., Insurance, Banking, Tech, Aerospace, etc.). If you dont know domain fill "Any".

    Please format your response in YAML as follows:

    ```yaml
    Requirements:
    - requirement1
    - ...
    Domain: domain_name
    ```

    Job Description
    {}

    '''.format(data)

    # Calling Gemini API to get response
    response, _ = util.gemini_prompt_call(text)
    output = response.text
    
    # Extracting requirements and domain from Gemini response
    req_start = output.index("Requirements:")
    domain_start = output.index("Domain:")
    final_end = output.index("\n", domain_start) if "\n" in output[domain_start:] else len(output)
    
    return output[req_start:domain_start], output[domain_start:final_end]


def get_jd_yoe(data):
    """
    Function to extract years of experience required from a job description.

    Args:
    - data (str): Job description text.

    Returns:
    - int: Years of experience required.
    """
    pattern = r'\d+'
    numbers = re.findall(pattern, data)
    yoe = int(max(numbers))
    return yoe


def get_jd_domain_reqs_stop_words(data):
    """
    Function to extract requirements from a job description while ignoring stop words.

    Args:
    - data (str): Job description text.

    Returns:
    - str: Extracted requirements text.
    """
    from nltk.corpus import stopwords

    pattern = r'\d+'
    numbers = re.findall(pattern, data)
    yoe = numbers, int(max(numbers))

    stop_words = set(stopwords.words('english'))
    ans = []

    for line in data.split("\n"):
        for word in re.split(';|,|\s|/', line):
            if word.lower() not in stop_words:
                ans.append(word)
    
    return " ".join(ans), ""


def get_jd_domain_yoe(data):
    """
    Function to extract domain and years of experience from a job description using Gemini.

    Args:
    - data (str): Job description text.

    Returns:
    - tuple: Extracted domains (list of strings) and years of experience (int).
    """
    # Constructing prompt for Gemini API call
    text = \
    '''
    Assume you are a senior IT recruiter with over 20 years of experience. Your task is to identify the domain or related sector based on the given job description (e.g., Insurance, Banking, Aerospace, etc.). Also, you are supposed to find out the required years of experience and return the findings in YAML format.

    ```yaml
    - Domains: Domain1, Domain2, Domain3
    - Years of Experience: Years

    # Job Description

    {}
    '''.format(data)

    # Calling Gemini API to get response
    response, _ = util.gemini_prompt_call(text)
    lines = response.text.lower().split("\n")

    # Finding lines containing domains and years of experience
    start, end = -1, -1
    for i in range(len(lines)):
        if string_util.check_word("domains", lines[i]):
            start = i
            continue
        elif start != -1 and string_util.check_word("years", lines[i]):
            end = i

    domains = "\n".join(lines[start:end]).strip().split(":")[1]
    yoe = lines[end].split(":")[1]

    return domains, max([int(ele) for ele in re.findall(r'\d+', yoe)])
