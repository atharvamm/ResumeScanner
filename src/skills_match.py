# # ---- Previous Attempt

# import re
# from nltk.corpus import stopwords
# from langchain_postgres import PGVector
# from src.process_text import get_doc_frm_st
# import numpy as np
# from src.experience import merge_overlapping_strings
# from openai import OpenAI
# import os

# '''
# # Keyword Search
# 1. Iteratively feed resume to LLM.
# 2. Iterate and only pick words not seen yet without changing order
# 3. RAG Query: Job Description Top 20 Matches.
# '''

# def jd_prompt(jd):
#     client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = "{}".format(os.getenv("NVIDIA_KEY"))
#     )

#     # text = '''
#     # Please review the job description below and identify the key points that should be sought in a candidate's resume. Return these key points as bullet points starting with '*':

#     # Job Description:
#     # {}

#     # Key Points: [Response]
#     # '''.format(jd)


#     # text = '''
#     # Please review the job description below and identify the key points to look for in a candidate's resume. Return these key points formatted as YAML.

#     # Job Description:
#     # {}

#     # Key Points:
#     # - [Sample Point 1]
#     # - [Sample Point 2]
#     # '''.format(jd)


#     # text = '''
#     # Please review the job description below and extract the required skills that should be looked for in a candidate's resume. Return these skills formatted as YAML.

#     # Job Description:
#     # {}

#     # Required Skills:
#     # - Technical Skill
#     # ...
#     # '''.format(jd)

#     text = '''
#     Please review the job description below and extract all required technical skills that candidates should highlight in their resumes. Return these skills formatted as YAML.

#     Job Description:
#     {}

#     Required Skills:
#     - 
#     '''.format(jd)


#     completion = client.chat.completions.create(
#     model="meta/llama2-70b",
#     messages=[{"role":"user","content":"{}".format(text)}],
#     temperature=0.0,
#     top_p=1,
#     max_tokens=1024,
#     stream=True
#     )
#     output = []
#     for chunk in completion:
#         if chunk.choices[0].delta.content is not None:
#             output.append(chunk.choices[0].delta.content)

#     ans = "".join(output).split("\n")
#     output = [ele for ele in ans if "- " in ele]
#     return "\n".join(output)



# def skills_prompt(jd, cur):
#     client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = "{}".format(os.getenv("NVIDIA_KEY"))
#     )

#     # text = '''
#     # Given the job description, the list of skills, and the missing skills from the previous resume section, analyze the text from the current resume section. Identify the matching skills and update the list of missing skills accordingly. Assume all skills are initially in the missing skills section and move them to the matching skills section as they appear in the current resume section.

#     # Job Description:
#     # {}

#     # Previous Response:
#     # {}

#     # Current Resume Section:
#     # {}

#     # Only include the answers to the questions; do not add any extra text.

#     # Matching Skills: [Response]
#     # Missing Skills: [Response]
#     # Percentage Matching: [Response]
#     # '''.format(jd, prev, cur)


#     # text = '''
#     # Given a job description, a list of required skills, and the skills missing from the previous resume section, your task is to analyze the text of the current resume section. Identify the skills that match those listed in the job description and update the list of missing skills accordingly. Assume that all skills are initially categorized as missing and move them to the matching skills section as they appear in the current resume section. Provide your response as bullet points starting with '*'.

#     # Job Description:
#     # {}

#     # Previous Resume Skills:
#     # {}

#     # Current Resume Section:
#     # {}

#     # Matching Skills: [Your Response]
    
#     # Missing Skills: [Your Response]
    
#     # Percentage Match: [Your Response]
#     # '''.format(jd, prev, cur)


#     # text = '''
#     # Given a job description, a list of previous resume skills, and a current resume section, your task is to:

#     # 1. Identify skills in the current resume section that match those in the job description.
#     # 2. Update the list of missing skills by moving missing skills to the matching skills section.
#     # 3. Calculate the percentage of job description skills found in the current resume section.

#     # Assume all skills are initially categorized as missing. Move them to the matching skills section as they appear in the current resume section.

#     # Return the response formatted as YAML with the following structure:

#     # Job Description:
#     # {}

#     # Previous Resume Skills:
#     # {}

#     # Current Resume Section:
#     # {}

#     # Matching Skills:
#     # - [Skill 1]
#     # - [Skill 2]

#     # Missing Skills:
#     # - [Skill 1]
#     # - [Skill 2]

#     # Percentage Match:
#     # - [Number]%
#     # ''' .format(jd, prev, cur) 

#     # text = '''
#     # Given the keypoints from the job description, the list of matched skills and missing skills from the previous section of the resume, and the current resume section, your task is to:

#     # 1. Identify skills in the current resume section that match those in the job description.
#     # 2. Add skills from the job description that are not in the matched skills section to the missing skills section.
#     # 3. Calculate the percentage of job description skills found in the matched skills section.

#     # Job Description:
#     # {}

#     # Previous Resume Skills:
#     # {}

#     # Current Resume Section:
#     # {}

#     # Return the response formatted as YAML:

#     # - Matching Skills:
#     #     - Skill 1

#     # - Missing Skills:
#     #     - Skill 1

#     # - Percentage Match: [Number]%
#     # '''.format(jd, prev, cur)


#     text = '''
#     Given the required skills and the resume section, identify the skills that are not in the resume section and add them to the missing skills. If the resume section contains skills that are also in the job description, add them to the matching skills.

#     Required Skills:
#     {}

#     Resume Section:
#     {}

#     Return the response formatted as YAML:

#     Matching Skills:
#     - 

#     Missing Skills:
#     - 
#     '''.format(jd, cur)

#     completion = client.chat.completions.create(
#     model="meta/llama2-70b",
#     messages=[{"role":"user","content":"{}".format(text)}],
#     temperature=0.0,
#     top_p=1,
#     max_tokens=1024,
#     stream=True
#     )
#     output = []
#     for chunk in completion:
#         if chunk.choices[0].delta.content is not None:
#             output.append(chunk.choices[0].delta.content)

#     return output



# def pick_set(text):
#     stop_words = set(stopwords.words('english'))
#     seen_words = set()
#     ans = []
#     # print(text)
#     for line in text.split("\n"):
#         # print(line)
#         for word in re.split(';|,|\s|/', line):
#             if word not in seen_words and word not in stop_words:
#                 seen_words.add(word)
#                 ans.append(word)
    
#     return " ".join(ans)


# def normalize_vectors(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     return vectors / norms

# def rag_skills(query, text):
#     # Get query embedding
#     # Then search similar vectors in vector store
#     # Send jd and resume parts to llm.
#     pass

# def rag_novectorstore_skills(query, text, threshold = 0.1):
#     # - https://github.com/langchain-ai/langchain/issues/2442
#     # Get query embedding
#     # Then search similar vectors and sort them select a particular score to filter to.
#     # Send jd and resume to llm. 

#     # k = 5
#     # query_vec = embeddings.embed_query("my_query")
#     # doc_vecs = np.array(embeddings.embed_documents(docs))
#     # scores = query_vec @ doc_vecs.T
#     # top_docs = [docs[i] for i in np.argsort(-scores)[:k]]
#     from src.vector_store import get_embedding
#     from src.parse_doc import text_splitter
#     added_docs = set()
#     ans = []


#     query_docs = text_splitter(get_doc_frm_st(query))
#     text_docs = text_splitter(get_doc_frm_st(text))

#     # query_embeddings = np.array([np.array(embedding) for embedding in get_embedding([query_doc.page_content for query_doc in query_docs])])
#     # text_embeddings = np.array([np.array(embedding) for embedding in get_embedding([text_doc.page_content for text_doc in text_docs])])

#     query_embeddings = np.array(get_embedding([query_doc.page_content for query_doc in query_docs]))
#     text_embeddings = np.array(get_embedding([text_doc.page_content for text_doc in text_docs]))

#     # print(query_embeddings.shape,text_embeddings.shape)
#     # print(query_embeddings[0].shape,text_embeddings[0].shape)

#     # print(type(query_embeddings),type(text_embeddings))
#     # print(type(query_embeddings[0]),type(text_embeddings[0]))

#     # for ind,doc in enumerate(text_docs):
#     #     print(ind,doc)
    
#     normalized_doc_vecs = normalize_vectors(text_embeddings)

#     for ind,query in enumerate(query_embeddings):
#         print("Query:",query_docs[ind])
#         normalized_query = normalize_vectors(query.reshape(1, -1))
#         # Compute cosine similarity scores
#         scores = normalized_query @ normalized_doc_vecs.T
#         print(scores)
#         for score_ind,score in enumerate(scores[0]):
#             if score > threshold:
#                 added_docs.add(score_ind)

#         # k = 5
#         # top_docs_indices = np.argsort(-scores[0])[:k]  # scores[0] because query_vec is 2D (1, n)
#         # print(top_docs_indices)
#         # top_docs = [text_docs[i] for i in top_docs_indices]

#         # Optionally, get the top scores
#         # top_scores = scores[0][top_docs_indices]

#         # Printing results
        
#     print("Final Docs:", added_docs)
#     for index in sorted(added_docs):
#         ans.append(text_docs[index].page_content)
#     return merge_overlapping_strings(ans)

# # 1. Set a threshold of 0 and select documents above that threshold.
# # 2. 




# # ------
# # Previous
# # ------


# # def check_non_numeric(s):
# #     for conv in (int, float, complex):
# #         try:
# #             conv(s)
# #             return False
# #         except ValueError:
# #             continue
# #     return True

# # def get_spacy(text):
# #     # >>> import spacy
# #     # >>> nlp = spacy.load("en_core_sci_lg")
# #     # >>> text = """spaCy is an open-source software library for advanced natural language processing, 
# #     # written in the programming languages Python and Cython. The library is published under the MIT license
# #     # and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
# #     # >>> doc = nlp(text)
# #     # >>> print(doc.ents)
# #     # (spaCy, open-source software library, written, programming languages,
# #     # Python, Cython, library, MIT, license, developers, Matthew Honnibal, 
# #     # Ines, Montani, founders, software company)

# #     import spacy
# #     nlp = spacy.load("en_core_web_lg")
# #     doc = nlp(text)
# #     # print(doc.ents)
# #     return doc.ents

# # def get_yake(text):
# #     # >>> import yake
# #     # >>> kw_extractor = yake.KeywordExtractor()
# #     # >>> text = """spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
# #     # >>> language = "en"
# #     # >>> max_ngram_size = 3
# #     # >>> deduplication_threshold = 0.9
# #     # >>> numOfKeywords = 20
# #     # >>> custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
# #     # >>> keywords = custom_kw_extractor.extract_keywords(text)
# #     # >>> for kw in keywords:
# #     # ...     print(kw)


# #     import yake
# #     # kw_extractor = yake.KeywordExtractor()
# #     language = "en"
# #     max_ngram_size = 4
# #     deduplication_threshold = 0.5
# #     numOfKeywords = 500
# #     custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
# #     keywords = custom_kw_extractor.extract_keywords(text)
# #     # for kw in keywords:
# #     #     print(kw)
# #     return keywords


# # def get_rake_nltk(text):
# #     # >>> from rake_nltk import Rake
# #     # >>> rake_nltk_var = Rake()
# #     # >>> text = """spaCy is an open-source software library for advanced natural language processing,
# #     # written in the programming languages Python and Cython. The library is published under the MIT license
# #     # and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
# #     # >>> rake_nltk_var.extract_keywords_from_text(text)
# #     # >>> keyword_extracted = rake_nltk_var.get_ranked_phrases()
# #     # >>> print(keyword_extracted)


# #     from rake_nltk import Rake
# #     rake_nltk_var = Rake()
# #     rake_nltk_var.extract_keywords_from_text(text)
# #     keyword_extracted = rake_nltk_var.get_ranked_phrases()
# #     return keyword_extracted


# # def check_keywords(keywords, wanted_keywords):
# #     keywords = set(keywords)
# #     for keyword in keywords:
# #         if keyword in wanted_keywords:
# #             return True
# #     return False


# # ---- Previous Attempt

'''
##################
Latest Attempt
##################
'''
from src.vector_store import get_embedding
from src.parse_doc import text_splitter
import numpy as np
from src.process_text import get_doc_frm_st



def find_overlap(a, b):
    """Find the maximum overlap between the end of string a and the start of string b."""
    max_overlap = 0
    overlap_str = ""
    
    # Check overlap from the end of a to the start of b
    for i in range(1, min(len(a), len(b)) + 1):
        if a[-i:] == b[:i]:
            max_overlap = i
            overlap_str = a + b[i:]
    
    return max_overlap, overlap_str

def merge_overlapping_strings(strings):
    """Merge overlapping strings into one string."""
    if not strings:
        return ""
    
    merged_string = strings[0]
    
    for i in range(1, len(strings)):
        max_overlap, overlap_str = find_overlap(merged_string, strings[i])
        if max_overlap > 0:
            merged_string = overlap_str
        else:
            merged_string += " " + strings[i]
    
    return merged_string


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def query_text_cosine_score(query_docs, text_docs, threshold = 0.0):
    # https://github.com/langchain-ai/langchain/issues/2442

    query_embeddings = np.array(get_embedding([query_doc.page_content for query_doc in query_docs]))
    text_embeddings = np.array(get_embedding([text_doc.page_content for text_doc in text_docs]))
    normalized_text_vecs = normalize_vectors(text_embeddings)

    total_score, query_length = 0,0
    for query in query_embeddings:
        # print("Query:",query_docs[ind])
        normalized_query = normalize_vectors(query.reshape(1, -1))
        # Compute cosine similarity scores
        scores = normalized_query @ normalized_text_vecs.T
        # print(query.shape,len(query))
        # print(np.mean(scores))
        query_length += len(query)
        # total_score += np.mean(scores)*len(query)
        total_score += np.max(scores)*len(query)

        # k = 5
        # top_docs_indices = np.argsort(-scores[0])[:k]  # scores[0] because query_vec is 2D (1, n)
        # print(top_docs_indices)
        # top_docs = [text_docs[i] for i in top_docs_indices]

        # Optionally, get the top scores
        # top_scores = scores[0][top_docs_indices]

        # Printing results
    
    return round(((total_score/query_length + 1)/2)*100,5)


def rag_novectorstore_skills(query, text, threshold = 0.0):

    added_docs = set()
    ans = []


    query_docs = text_splitter(get_doc_frm_st(query))
    text_docs = text_splitter(get_doc_frm_st(text))

    query_embeddings = np.array(get_embedding([query_doc.page_content for query_doc in query_docs]))
    text_embeddings = np.array(get_embedding([text_doc.page_content for text_doc in text_docs]))
    
    normalized_doc_vecs = normalize_vectors(text_embeddings)

    for ind,query in enumerate(query_embeddings):
        # print("Query:",query_docs[ind])
        normalized_query = normalize_vectors(query.reshape(1, -1))
        scores = normalized_query @ normalized_doc_vecs.T
        print(scores)
        for score_ind,score in enumerate(scores[0]):
            if score > threshold:
                added_docs.add(score_ind)

        # k = 5
        # top_docs_indices = np.argsort(-scores[0])[:k]  # scores[0] because query_vec is 2D (1, n)
        # print(top_docs_indices)
        # top_docs = [text_docs[i] for i in top_docs_indices]
        # top_scores = scores[0][top_docs_indices]
        
    print("Final Docs:", added_docs)
    for index in sorted(added_docs):
        ans.append(text_docs[index].page_content)
    return merge_overlapping_strings(ans)