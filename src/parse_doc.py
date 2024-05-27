import os
from langchain_community.document_loaders import PyPDFLoader,PDFMinerLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from src.load_env import get_root_path
from pdfminer.high_level import extract_text
from langchain.schema.document import Document
import pdfplumber

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    for i,text in enumerate(texts):
        text.metadata["doc_seq"] = i + 1
    return texts

def parse_pdf(doc_path):
    # print(doc_path)
    # loader = PyPDFLoader(doc_path)
    # loader = PDFMinerLoader(doc_path)
    # data = loader.load()
    # string = extract_text(doc_path)

    pdf = pdfplumber.open(doc_path)
    pages = []

    for page in pdf.pages:
        pages.append(page.extract_text())

    string = "\n".join(pages)
    data =  [Document(page_content=string, metadata={"source": "{}".format(doc_path)})]
    # return text_splitter(data)
    return data

def parse_docx(doc_path):
    # print(doc_path)
    loader = Docx2txtLoader(doc_path)
    data = loader.load()
    # return text_splitter(data)
    return data

def parse_doc(doc_path):
    loader = UnstructuredWordDocumentLoader(doc_path,mode = "single")
    data = loader.load()
    # return text_splitter(data)
    return data

def parse_dir(path):
    doc_funcs = {
        "doc" : parse_doc,
        "docx" : parse_docx,
        "pdf" : parse_pdf
    }

    docs_data = []
    doc_list = os.listdir(path)  
    for doc_name in doc_list:
        if "DS_Store" in doc_name or "~" == doc_name[0]:
            continue
        doc_path = os.path.join(path,doc_name)

        if "__" in doc_name[:2]:
            # data = ""
            continue

        ext = doc_name.split(".")[-1]
        data = doc_funcs[ext](doc_path)

        # elif ".doc" in doc_name[-6:]:
        #     data = parse_doc(doc_path)
            
        # elif ".pdf" in doc_name[-6:]:
        #     pdf_path = os.path.join(path,doc_name)
        #     data = parse_pdf(pdf_path)
        docs_data.extend(data)
    
    # print("Current: ",type(docs_data),type(docs_data[0]))
    return docs_data



# # ----
# # Table
# # ----

# pdf_path = "/Users/atharvamhaskar/Documents/ResumeScanner/docs/projects/cyberark/JD/Sailpoint-CyberArk-OFFSHORE-CHENNAI-job-description.pdf"

# import pdfplumber
# from operator import itemgetter

# def check_bboxes(word, table_bbox):
#     """
#     Check whether word is inside a table bbox.
#     """
#     l = word['x0'], word['top'], word['x1'], word['bottom']
#     r = table_bbox
#     return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

# with pdfplumber.open(pdf_path) as pdf:
#     pages = []
#     for page in pdf.pages:
#         page.extract_text()

#         tables = page.find_tables()
#         table_bboxes = [i.bbox for i in tables]
#         tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
#         non_table_words = [word for word in page.extract_words() if not any(
#             [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
#         lines = []
#         for cluster in pdfplumber.utils.cluster_objects(
#                 non_table_words + tables, itemgetter('top'), tolerance=5):
#             if 'text' in cluster[0]:
#                 lines.append(' '.join([i['text'] for i in cluster]))
#             elif 'table' in cluster[0]:
#                 lines.append(cluster[0]['table'])
#         pages.append(lines)
# print(pages)