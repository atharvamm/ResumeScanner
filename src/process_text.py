import re
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.schema.document import Document


def clean_text(text):
    text = re.sub(r'[^\w\s+-.,\'\/]', '', text)
    # text = re.sub(r'\n+', '\n', text)
    # text = re.sub(r'\s+', ' ', text)
    return text


def process_text(text):
    text = text.lower()
    text = clean_text(text)
    tokens = text.split(" ")
    return tokens
    # return cleaned_tokens


def get_doc_frm_st(string = "", source = "" ):
    return [Document(page_content = string, metadata = {"source": "{}".format(source)})]