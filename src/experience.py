
# # ----- Previous Attempt

# # 1. Code to merge intervals
# # 2. Code to find total experience
# # 3. Code to find gaps in experience
# # 4. Code to find experience start and end as list from resume.
# from openai import OpenAI
# import os
# from src.process_text import process_text
# from src.parse_doc import text_splitter
# from src.vector_store import get_vectorstore
# from nltk.corpus import stopwords

# def experience_all(data):
#     import re
#     import datetime
    
#     regex_strings = [
#         r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:'|\s*)(?:\d{4}|\d{2})",
#     ]

#     data = data.split(" ")
#     for i,ele in enumerate(data):
#         if "-" in ele:
#             x = " ".join(data[i - 5: i + 5]).lower()
#             print("\n",x,end = " ")
#             for regex_string in regex_strings:
#                 results = re.findall(regex_string,x)
#                 if ("present".lower() in x or "till date".lower() in x) and len(results) == 1:
#                     results.append(datetime.datetime.now().date())
#                 if len(results) > 0:
#                     print(results)
#                 if len(results) > 1:
#                     break


# import re
# date_pattern = re.compile(
#     r"""
#     (?:
#         \b(?:0?[1-9]|[12][0-9]|3[01])         # Match day (01-31)
#         ([-/\s.])                             # Separator
#         (?:0?[1-9]|1[0-2])                    # Match month (01-12)
#         \1                                    # Repeat the separator
#         (?:\d{2,4})\b                         # Match year (yy or yyyy)
#     |
#         \b(?:0?[1-9]|[12][0-9]|3[01])         # Match day (01-31)
#         ([-/\s.'])                            # Separator
#         (?:\d{2})\b                           # Match year (yy)
#     |
#         \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)  # Match abbreviated month names
#         (?:uary|ruary|ch|il|e|y|e|ust|ober|ember)?     # Match full month names
#         \s?`?'?\d{2,4}                         # Match year with optional apostrophe (yy or yyyy)
#         (?:\b|[\s-])
#     |
#         \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)  # Match abbreviated month names
#         \s?'?\d{2,4}\b                         # Match year with optional apostrophe (yy or yyyy)
#     |
#         \b\d{4}\s*[-\s]\s*\d{4}\b
#     )
#     """, re.VERBOSE | re.IGNORECASE)

# def contains_date(text):
#     return bool(date_pattern.search(text))


# def find_overlap(a, b):
#     """Find the maximum overlap between the end of string a and the start of string b."""
#     max_overlap = 0
#     overlap_str = ""
    
#     # Check overlap from the end of a to the start of b
#     for i in range(1, min(len(a), len(b)) + 1):
#         if a[-i:] == b[:i]:
#             max_overlap = i
#             overlap_str = a + b[i:]
    
#     return max_overlap, overlap_str

# def merge_overlapping_strings(strings):
#     """Merge overlapping strings into one string."""
#     if not strings:
#         return ""
    
#     merged_string = strings[0]
    
#     for i in range(1, len(strings)):
#         max_overlap, overlap_str = find_overlap(merged_string, strings[i])
#         if max_overlap > 0:
#             merged_string = overlap_str
#         else:
#             merged_string += " " + strings[i]
    
#     return merged_string


# from dateutil.parser import parse
# from dateutil.parser._parser import ParserError

# def is_date(string, fuzzy=True):
#     """
#     Return whether the string can be interpreted as a date.

#     :param string: str, string to check for date
#     :param fuzzy: bool, ignore unknown tokens in string if True
#     """
#     try: 
#         parse(string, fuzzy=fuzzy)
#         return True

#     except Exception as e:
#         if e in [OverflowError,ValueError,ParserError]:
#             return False
#         # else:
#             # print(e)

# def has_numbers(inputString):
#     return any(char.isdigit() for char in inputString)

# def clean_exp_llm(data):
#     lines = data.split("\n")
#     strings = []
#     for i in range(len(lines)):

#         start,end = 0,len(lines[i])
#         while start < end:
#             last_index = min(start + 150,end)
#             cur_str = lines[i][start:last_index]
#             # print("Cur String: ",cur_str)
#             if contains_date(cur_str):
#                 # print("Adding: ", cur_str)
#                 strings.append(cur_str)
#             elif has_numbers(cur_str[-25:]):
#                 cur_str = lines[i][start:min(last_index + 51,end)]
#                 # print("Cur String: ",cur_str)
#                 if contains_date(cur_str):
#                     # print("Adding: ", cur_str)
#                     strings.append(cur_str)
#             start += 100

#     bef,aft = lines[max(i-2,0):i],lines[i+1:min(i+3,len(lines) - 1)]

#     # if "," in bef:
#     #     bef =  bef.split(",")[-1]
#     # if "," in aft:
#     #     aft =  aft.split(",")[0]

#     # print("Before:",bef)
#     # print("After:",aft)
#     strings = bef + strings
#     strings.extend(aft)
#     # strings.append("\n\n")

#     string = merge_overlapping_strings(strings)
#     return string


# def experience_llm(text):
#     client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = "{}".format(os.getenv("NVIDIA_KEY"))
#     )
    
#     # text = "{}\n\n Extract the beginning, end, and organization of experiences from the resume and provide them as a list of tuples (Begin,End, Org). Output only the list and nothing else.".format(text)

#     text = '''
#     Below is my resume. Please analyze it and provide the following information:
#     1. Total Experience: Calculate the total number of years and months of work experience.
#     2. Employment Gaps: Identify any significant employment gaps (gaps of more than 3 months) and provide the duration of each gap.
#     3. Average Tenure: Calculate the average tenure of my jobs in years and months.
#     4. Explanation: Explain how you calculated the total experience and identified the employment gaps.
#     \n\n
#     Resume:
#     {}

#     Format your response as follows:
#     Total Experience: <Answer>,
#     Employment Gaps: <Answer>,
#     Average Tenure: <Answer>,
#     Explanation: <Answer>
#     '''.format(text)

#     completion = client.chat.completions.create(
#     model="meta/llama2-70b",
#     messages=[{"role":"user","content":"{}".format(text)}],
#     temperature=0.2,
#     top_p=0,
#     max_tokens=1024,
#     stream=True
#     )
#     output = []
#     for chunk in completion:
#         if chunk.choices[0].delta.content is not None:
#             output.append(chunk.choices[0].delta.content)

#     return output
    





# def experience_rag(data):
#     data.page_content = process_text(data.page_content)
#     texts = text_splitter(data)

#     vectorstore = get_vectorstore("resume_coll",new_conn = True)
#     vectorstore.add_documents(texts)
#     results = vectorstore.similarity_search_with_score("What are the start and end date of experiences ? ", k = 5)




#     pass


# '''
# # get_connection()
# # vectorstore = get_vectorstore("resume_coll",new_conn = True)
# # vectorstore.add_documents(docs_data)

# vectorstore = get_vectorstore("resume_coll",new_conn = False)

# results = vectorstore.similarity_search_with_score(job_description, k = 5)

# # results = vectorstore.max_marginal_relevance_search_with_score(job_description,k = 20, fetch_k=20,lambda_mult=0)

# for result in results:
#     # print(result)
#     print(result[0].metadata["source"])
#     # print(result)
#     # print("\n"*2)

# # with open(os.path.join(get_root_path(),"dump","embedding.txt"),"w") as f:
# #     f.write(str(get_embedding(job_description)[0]))
# # print("Final!!!")


# ## Temp

# # Score = print(vectorstore.similarity_search_with_score("machine learning", k = 3)[0][1])

# '''



# if __name__ == "__main__":
#     with open("/Users/atharvamhaskar/Documents/ResumeScanner/text.txt","r") as f:
#         data = f.read()
#     experience_all(data)

# '''

# # ----

# # 1. Pattern 1 : String followed by number
# import re

# text = "Write regex pattern to match: Jan'20,Jan '20, Jan '20,January '20, March '20, Jan 2020, November2020"

# patterns = re.findall(r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*(?:'|\s*)(?:\d{4}|\d{2})", text)

# print(patterns)


# # 



# '''


# # - Tried pdf plumber to pull date strings
# # -




# # ['', ' ', " (Sep '22, Present, Shopse)\n(Jun '20, Jul '22, Ascent Cyber Sol", "utions)\n(Jun '19, Jun '20, Avataar.ai)\n(Jul '19, Jun '20", ", Quantiphi)\n(Jul '19, Jun '20, Embibe - Ascent Cyber Solutions)\n(Jul '", "19, Jun '20, Trace VFX[Technicolor])\n(Sep '18, May '19, Mob", 'ily', 'te[Infostride])', '']


# # ['', ' ', ' (MARCH 2024, PRESENT, STRATASYS)\n(FEBRUARY 2022, MAR', 'CH 2024, INFOGAIN PVT LTD)\n(DECEMBER 2020, JANUARY', ' 2022, PERSISTENT SYSTEM LTD)\n(APRIL 2017, DECEMBER', ' ', '2020, SENSEGIZ TECHNOLOGY)', '']


# # ----- Previous Attempt

'''
##################
Latest Attempt
##################
'''
import numpy as np
from src.parse_doc import text_splitter
from src.vector_store import get_embedding
import src.util as util
import src.string_util as string_util
from datetime import datetime,date
from openai import OpenAI
import os
import re
import time
import google.generativeai as genai
from thefuzz import fuzz
from dateutil.relativedelta import relativedelta

scores = {
    "none" : 0.05,
    "low" : 0.33,
    "medium" : 0.67,
    "high" : 1
}

def get_score(key):
    global scores
    pattern = r'\b(none|low|medium|high)\b'
    # print("Input:",key,"Matches:",re.findall(pattern, key))
    key = re.findall(pattern, key)[0]
    key = key.strip()
    return scores.get(key,0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_baseline_score(yoe):
    total_score = sigmoid(0)
    # duration_weight = 0.9
    duration_weight = 0.99**(yoe/2)

    technical_exp = scores["medium"] * yoe * duration_weight
    technical_score = sigmoid(technical_exp - yoe)

    domain_exp = scores["low"] * yoe * duration_weight
    domain_score = sigmoid(domain_exp - yoe)

    softskill_exp = scores["medium"] * yoe * duration_weight
    softskill_score = sigmoid(softskill_exp - yoe)
    print("YOE,Weight,Technical Exp,Domain Exp,Softskill Exp:",yoe,duration_weight,technical_exp,domain_exp,softskill_exp)
    print("Baseline:Total,Technical,Domain,Softskill:",total_score,technical_score,domain_score,softskill_score)

    final_score = 0.25*total_score + 0.55*technical_score + 0.05*softskill_score + 0.15*domain_score
    print("Expected Final Baseline:",final_score)

    baseline = {
        "total_exp" : yoe,
        "technical_exp" : technical_exp,
        "domain_exp" : domain_exp,
        "softskill_exp" : softskill_exp,

        "total_score" : round(total_score,3),
        "technical_score" : round(technical_score,3),
        "domain_score" : round(domain_score,3),
        "softskill_score" : round(softskill_score,3),
        "final_score" : round(final_score,3)
    }
    return baseline

def get_experience_yaml(data):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "{}".format(os.getenv("NVIDIA_KEY"))
    )

    # text = '''
    # Resume Section
    # {}

    # Find the "Experience" section in the resume and return the work experience details in YAML format. The dates should be in MM/YYYY format. If the end date is present as "till date", "now", etc., include the string as it is. The "Technical" field should be a boolean (True or False). The "Organization Domain" should describe the industry, such as Finance, IT Consultancy, etc.

    # Return the following YAML structure for each work experience entry:
    # - Work Experience:
    #     - Organization: 
    #         - StartDate: 
    #         - EndDate: 
    #         - Title: 
    #         - Technical: 
    #         - Organization Domain: 
    # '''.format(data)


    text = '''
    Resume Section
    {}

    Extract the work experiences from the resume and return the details in YAML format. Follow these guidelines:

    - If the end date is not a specific date but a string (e.g., "present", "till date", "now"), include the string "present".
    - The "Technical" field should be a boolean, indicating whether the position was technical (e.g., software, IT-related) or not.

    Return the following YAML structure for each work experience entry:

    WorkExperience:
    - Organization:
        - StartDate: 
        - EndDate: 
        - Title: 
        - Technical: 
    '''.format(data)

    completion = client.chat.completions.create(
    model="meta/llama2-70b",
    messages=[{"role":"user","content":"{}".format(text)}],
    temperature=0.0,
    top_p=0.00,
    max_tokens=1024,
    stream=True
    )
    output = []
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            output.append(chunk.choices[0].delta.content)

    return output

def scan_all_experience(cur_data,yoe):
    college_words = ["college","university"]
    lines = [ele for ele in cur_data.strip().split("\n") if len(ele) > 0]
    i = 0
    info = []

    while i < len(lines):
        # if "- Organization" in lines[i]:
        # if check_organization(lines[max(0,i-1)]+lines[i]) and len(lines[i].strip().split(":")[-1].strip()) > 0:
        if string_util.check_organization(lines[max(0,i-1)]+lines[i]):
            # print(lines[i:i+6])
            org = lines[i].strip().split(":")[-1].strip()
            start_date = lines[i+1].strip().split(":")[-1].strip()
            end_date = lines[i+2].strip().split(":")[-1].strip()

            # print(lines[i+3:i+6])
            # print("OG Technical:",lines[i+4],"Input Technical:",lines[i+4].strip().split(":")[1].strip().lower())
            # print("OG Domain:",lines[i+5],"Input Domain:",lines[i+5].strip().split(":")[1].strip().lower())

            technical_score = get_score(lines[i+4].strip().split(":")[1].strip().lower())
            domain_score = get_score(lines[i+5].strip().split(":")[1].strip().lower())
            softskill_score = get_score(lines[i+5].strip().split(":")[1].strip().lower())
            # print(lines[i+4].strip().split(":")[-1].strip().lower())
            # print("org:",org,"start:",start_date,"end:",end_date)    
            # print("technical score:",technical_score,"domain score:",domain_score)
            if len(start_date) > 1:
                info.append([org,technical_score,domain_score,softskill_score,start_date,end_date])
            elif end_date.lower() == "present":
                start_date = str(datetime.today().date())
                info.append([org,technical_score,domain_score,softskill_score,start_date,end_date])
            else:
                i += 1
                continue
            for word in college_words:
                if word in org.lower():
                    info.pop()
                    break
            
            i += 2
        i += 1

    # print(info)
    for i in range(len(info)):
        # print(ele)
        if info[i][-1].lower() == "present":
            ans = "present"
        else:
            ans = util.parse_date(info[i][-1])

        info[i].append(util.parse_date(info[i][-2]))
        info[i].append(ans)



    info.sort(key = lambda x : x[-2])

    for i in range(len(info)):
        # print(info[i])
        if info[i][-2] == datetime.today().date():
            info[i][-2] = info[i - 1][-1]
        if info[i][-1] == "present":
            if i == len(info) - 1:
                info[i][-1] = datetime.today().date()
            else:
                info[i][-1] = info[i + 1][-2]

    info = string_util.merge_experience_list(info)
    # print(info)

    total_exp,technical_exp,domain_exp,softskills_exp = 0,0,0,0
    # technical_wt,domain_wt,softskills_wt = 0,0,0

    i = 0
    while i < len(info):
        cur_exp = util.diff_in_years(start_date = info[i][-2], end_date = info[i][-1])
        if cur_exp > 0 :
            total_exp += cur_exp
            technical_exp += info[i][1] * cur_exp
            domain_exp += info[i][2] * cur_exp
            softskills_exp += info[i][3] * cur_exp

            # total_exp += cur_exp
            # technical_exp += info[i][1] * cur_exp
            # technical_wt += info[i][1]
            # domain_exp += info[i][2] * cur_exp
            # domain_wt += info[i][2]
            # softskills_exp += info[i][3] * cur_exp
            # softskills_wt += info[i][3]            

            # total_exp += cur_exp
            # technical_exp += info[i][1]
            # domain_exp += info[i][2]
            # softskills_exp += info[i][3]           
            i += 1
        else:
            info.pop(i)
    # technical_score = ((technical_exp/technical_wt)/total_exp)
    # domain_score = ((domain_exp/domain_wt)/total_exp)
    # softskill_score = ((softskills_exp/softskills_wt)/total_exp)

    # technical_score = (technical_exp/len(info))
    # domain_score = (domain_exp/len(info))
    # softskill_score = (softskills_exp/len(info))

    # technical_score = technical_exp
    # domain_score = domain_exp
    # softskill_score = softskills_exp
    # total_score = 1/(1 + np.exp(-(total_exp - yoe)))

    print("Total,Technical,Domain,Softskills:{:.3f},{:.3f},{:.3f},{:.3f}".format(total_exp,technical_exp,domain_exp,softskills_exp))
    technical_score = sigmoid(technical_exp - yoe)
    domain_score = sigmoid(domain_exp - yoe)
    softskill_score = sigmoid(softskills_exp - yoe)
    total_score = sigmoid(total_exp - yoe)

    # print(total_exp,yoe,(total_exp - yoe),1/(1 + np.exp(-(total_exp - yoe))))
    # total_score = 1/(1 + np.exp(-(total_exp - yoe)))
    return info,round(total_score,3),round(technical_score,3),round(domain_score,3),round(softskill_score,3)


def get_experience_report(jd,data):
    import time
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "{}".format(os.getenv("LLAMA3"))
    )

    text = '''
    # Prompt

    Extract the work experiences from the resume and return the details in YAML format. Follow these guidelines:

    - If the end date is not a specific date but a string (e.g., "present", "till date", "now"), include the string "present".

    - Compare the experience at the organization (including all projects) with the job description. Provide a rating of [None], [Low], [Medium], or [High] based on the relevance.

    - Identify the domain of the projects at the organization (e.g., Insurance, Banking, Tech, Aerospace, etc.). Compare it with the job description domain and provide a rating of [None], [Low], [Medium], or [High].

    Return the following YAML structure for each work experience entry:


    WorkExperience:

    - Organization:

    - StartDate:

    - EndDate:

    - Title:

    - Match Score:

    - Domain Score:

    
    # Job Description
    {}

    # Resume Section
    {}
    '''.format(jd,data)

    output = []
    completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[{"role":"user","content":"{}".format(text)}],
    temperature=0.0,
    top_p=0.00,
    max_tokens=2048,
    seed=42,
    stream=True
    )
    output = []
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            output.append(chunk.choices[0].delta.content)

    time.sleep(int(os.getenv("SHORT_TIME")))
    return output


def get_experience_gemini(jd,data,model_name = None):
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]
    genai.configure(api_key=os.environ["GEMINI"])
    model = genai.GenerativeModel(model_name)
    # info = genai.get_model("models/"+model_name)
    # print((info.input_token_limit, info.output_token_limit))


    # text = '''
    # # Prompt

    # Extract the work experiences from the resume and return the details in YAML format. Follow these guidelines:

    # - If the end date is not a specific date but a string (e.g., "present", "till date", "now"), include the string "present".

    # - Compare the experience at the organization (including all projects) with the job description. Provide a rating of [None], [Low], [Medium], or [High] based on the relevance.

    # - Identify the domain of the projects at the organization (e.g., Insurance, Banking, Tech, Aerospace, etc.). Compare it with the job description domain and provide a rating of [None], [Low], [Medium], or [High].

    # Return the following YAML structure for each work experience entry:


    # WorkExperience:

    # - Organization:

    # - StartDate:

    # - EndDate:

    # - Title:

    # - Match Score:

    # - Domain Score:

    
    # # Job Description
    # {}

    # # Resume Section
    # {}
    # '''.format(jd,data)

    # text = '''
    # # Prompt

    # Extract the work experiences from the resume and return the details in YAML format. Follow these guidelines:

    # 1. **End Date Handling**:
    #     - If the end date is a string (e.g., "present", "till date", "now"), represent it as "present".

    # 2. **Relevance Rating**:
    #     - Compare the work experience (including all projects) at each organization with the job description.
    #     - Provide a relevance rating as one of the following: `[None]`, `[Low]`, `[Medium]`, or `[High]`.

    # 3. **Domain Identification and Rating**:
    #     - Identify the domain of the projects at each organization (e.g., Insurance, Banking, Financial, Fintech, Tech, Aerospace, etc.).
    #     - Determine the domain of the job description and provide a domain relevance rating as one of the following: `[Low]`, `[Medium]`, or `[High]`.

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - StartDate:
    # - EndDate:
    # - Title:
    # - MatchScore:
    # - DomainScore:
    # ```

    # # Job Description
    # {}
    # # Resume Section
    # {}  
    # '''.format(jd,data)

    # text = \
    # '''    
    # # Prompt

    # Extract the work experiences from the resume and return the details in YAML format. Follow these guidelines:

    # - If the end date is not a specific date but a string (e.g., "present", "till date", "now"), include the string "present".
    # - Compare the experience at each organization (including all projects) with the job description. Provide a technical rating of [None], [Low], [Medium], or [High] based on the alignment of the experience with the job description.
    # - Determine a list of suitable domains for the job description (e.g., Insurance, Banking, Financial, Fintech, Tech, Aerospace, etc.).
    # - Identify the domains of the projects at each organization and provide a domain relevance rating as one of the following: [Low], [Medium], or [High].

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - StartDate:
    # - EndDate:
    # - Title:
    # - TechnicalScore:
    # - DomainScore:
    # - TechnicalScoreExplanation:
    # - DomainScoreExplanation:
    # ```

    # # Job Description
    # {}

    # # Resume Section
    # {}
    # '''.format(jd,data)

    # text = \
    # '''
    # As a seasoned recruiter with over two decades of experience in the IT industry, you've successfully filled numerous positions across various levels for diverse client organizations. Now, I have a task for you:

    # ## Task

    # Extract work experiences from resumes and present the details in YAML format, following these guidelines:

    # 1. **Handling End Dates:**
    #     - If the end date is informal (e.g., "present", "till date", "now"), use "present" in the output.

    # 2. **Technical Rating:**
    #     - Evaluate the candidate's experience at each organization relative to the provided job description.
    #     - Assign a technical score rating of [None], [Low], [Medium], or [High] based on the alignment of their technical knowledge with the job description.
    #     - [High] indicates most of the required skills are present, [None] means none of the required skills are present.

    # 3. **Domain Relevance:**
    #     - Identify suitable domains for the job description (e.g., Insurance, Banking, Financial, Fintech, Tech, Aerospace, etc.).
    #     - Assess the projects undertaken and the domain of the organization to identify alignment with the job description's domains.
    #     - Provide a domain score of [Low], [Medium], or [High] based on specific evaluation criteria. Focus solely on the domain. Clarify factors considered in determining domain relevance.
    #     - [High] indicates experience directly related or having a direct application to the job description's domains, while [Low] indicates no relevance to the job description's domains.

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - StartDate:
    # - EndDate:
    # - Title:
    # - TechnicalScore:
    # - DomainScore:
    # - DomainScoreExplanation:
    # ```

    # # Job Description
    # {}

    # # Resume
    # {}
    # '''.format(jd, data)


    text = \
    '''
    As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # Task

    Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    1. **Handling End Dates:**
        - If the end date is informal (e.g., "present", "till date", "now"), use "present" in the output.

    2. **Technical Rating:**
        - Assess the candidate's experience at each organization relative to the provided job description.
        - Assign a technical score rating of [None], [Low], [Medium], or [High] based on the alignment of their technical skills with the job description.

    3. **Domain Relevance:**
        - Identify application domains for the individual projects.(e.g., Insurance, Banking, Fintech, etc.). 
        - Provide a domain score of [Low], [Medium], or [High] based on experience alignment with job description domains. Focus solely on project domains, ignore technical details.
    4. Rate soft - skills based on job description [Low], [Medium], or [High]


    Return the following YAML structure for each work experience entry:

    ```yaml
    WorkExperience:
    - Organization:
    - StartDate:
    - EndDate:
    - Title:
    - TechnicalScore:
    - DomainScore:
    - DomainScoreExplanation:
    - SoftSkillsScore:
    ```

    # Job Description

    {}

    # Resume Section

    {}
    '''.format(jd,data)

    num_tokens = model.count_tokens(text)
    gen_config = {
        "temperature": 0.0,
        "top_p": 0.00,
        "top_k": 1
    }

    response = model.generate_content(text,generation_config=gen_config)

    time.sleep(int(os.getenv("SHORT_TIME")))
    return response,num_tokens

def get_gemini_single_prompt(jd,data,model_name = None):
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]
    genai.configure(api_key=os.environ["GEMINI"])
    model = genai.GenerativeModel(model_name)
    # info = genai.get_model("models/"+model_name)
    # print((info.input_token_limit, info.output_token_limit))

    text = \
    '''
    Today is 8th June 2024
    As an experienced recruiter with over two decades in the IT industry, having successfully placed numerous candidates across various levels in diverse client organizations, I have a task for you:

    # Task

    Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    1. Calculate the total experience of the candidate. Provide an explanation by listing out a series of tuples in the format (Org Name, Start Date, End Date, Title).
    2. Assess the alignment of technical skills and knowledge in the job description with those in the resume, assigning a rating of [None], [Low], [Medium], or [High]. Dont just lookup keywords but judge the complexity of projects for the required years of experience or seniority level.
    3. Identify the application industry (e.g., Insurance, Banking, Aerospace, etc.) mentioned in the job description. Evaluate the alignment of the application industry of projects in the resume, providing a rating of [None], [Low], [Medium], or [High]. Even minor projects in the industry count, so be liberal. Explain the rating by listing out relevant projects.
    4. Evaluate the alignment of expected soft skills in the job description with those evident in the resume, assigning a rating of [None], [Low], [Medium], or [High].

    Return the following YAML structure for each work experience entry:

    ```yaml
    - Total Experience:
    - Experience Explanation: 
    - Technical Skills Match:
    - Technical Skills Explanation: 
    - Domain Experience:
    - Domain Experience Explanation:
    - Soft Skills Rating:
    ```

    # Job Description

    {}

    # Resume Section

    {}
    '''.format(jd,data)

    num_tokens = model.count_tokens(text)
    gen_config = {
        "temperature": 0.0,
        "top_p": 0.00,
        "top_k": 1
    }

    response = model.generate_content(text,generation_config=gen_config)

    time.sleep(int(os.getenv("SHORT_TIME")))
    return response,num_tokens

def get_dates_prompt(jd,data):
    text = \
    '''
    As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Your expertise is now needed to assist in structuring candidate work experiences in a standardized format.

    # Task

    Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    1. **Handling End Dates**: If the end date is informal (e.g., "present", "till date", "now"), use "present" in the output.
    2. **Rate Soft Skills**: Based on the job description provided, rate the soft skills as [Low], [Medium], or [High]. Consider attributes like communication, teamwork, problem-solving, and adaptability.

    ### YAML Structure

    For each work experience entry, return the following YAML structure:

    ```yaml
    WorkExperience:
    - Organization: 
    StartDate: 
    EndDate: 
    Title: 
    SoftSkillsScore: 
    ```

    # Job Description

    {}

    # Resume Section

    {}
    '''.format(jd,data)

    response,num_tokens = util.gemini_prompt_call(text)
    return response,num_tokens


def get_match_prompt(jd,data):
    # text = \
    # '''
    # As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # # Task

    # Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    # ### Guidelines:

    # 1. **Technical Rating:**
    # - Assess the candidate's experience at each organization relative to the provided job description. Evaluate if the complexity of the projects or leadership experience in a particular role aligns with the seniority and skillset expected for the job.
    # - Assign a technical score rating of [None], [Low], [Medium], or [High] based on this alignment.

    # 2. **Domain Relevance:**
    # - Identify the application domains for each experience (e.g., Insurance, Banking, Fintech).
    # - Provide a domain score of [Low], [Medium], or [High] based on the relevance of the candidate's experience to any of the job description domains. Focus solely on project domains, ignoring technical skills for this score.

    # ### Output Format:

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - Title:
    # - TechnicalScore:
    # - DomainScore:
    # - TechnicalScoreExplanation:
    # - DomainScoreExplanation:
    # ```

    # # Job Description

    # {}

    # # Resume Section

    # {}
    # '''.format(jd,data)

    # text = \
    # '''
    # As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # # Task

    # Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    # ### Guidelines:

    # 1. Technical Rating:
    # - Assess the candidate's experience at each organization relative to the job description.
    # - Assign a technical score of [None], [Low], [Medium], or [High] based on the following criteria:
    # - None: No relevant technical experience.
    # - Low: Low technical complexity.
    # - Medium: Moderate technical complexity or some leadership.
    # - High: High technical complexity or significant leadership.

    # 2. Domain Relevance:
    # - Identify the application domains for each experience (e.g., Insurance, Banking, Fintech).
    # - Assign a domain score of [Low], [Medium], or [High] based on the following criteria:
    # - Low: No relevance to job description domains.
    # - Medium: Moderate relevance to job description domains.
    # - High: High relevance to job description domains.

    # ### Output Format:

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - Title:
    # - TechnicalScore:
    # - DomainScore:
    # - TechnicalScoreExplanation:
    # - DomainScoreExplanation:
    # ```

    # # Job Description

    # {}

    # # Resume Section

    # {}
    # '''.format(jd, data)


    # text = \
    # '''
    # As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # # Task

    # Extract work experiences from resumes and present the details in YAML format, adhering to these guidelines:

    # ### Guidelines:

    # 1. Technical Rating:
    # - Assess the candidate's experience at each organization relative to the job description.
    # - Assign a technical score of [None], [Low], [Medium], or [High].
    # - None: No relevant technical experience.
    # - Low: Limited technical complexity.
    # - Medium: Moderate technical complexity or some leadership.
    # - High: High technical complexity or significant leadership.

    # 2. Domain Relevance:
    # - Identify the application domains for each experience (e.g., Insurance, Banking, Fintech).
    # - Assign a domain score of [Low], [Medium], or [High].
    # - Low: No relevance to job description domains.
    # - Medium: Moderate relevance to job description domains.
    # - High: High relevance to job description domains.

    # ### Output Format:

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization:
    # - Title:
    # - TechnicalScore:
    # - DomainScore:
    # ```

    # # Job Description

    # {}

    # # Resume Section

    # {}
    # '''.format(jd, data)

    # text = \
    # '''
    # As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # # Task

    # Extract work experiences from the resume provided and present the details in YAML format, adhering to these guidelines:

    # ## Guidelines:

    # 1. Technical Rating:
    # - Assess the candidate's technical experience at each organization relative to the job description.
    # - Assign a technical score of [None], [Low], [Medium], or [High] based on the following criteria:
    #     - None: 
    #     - No relevant technical experience.
    #     - Example: Worked in a non-technical role or unrelated field.
    #     - Low:
    #     - Involvement in low-complexity technical tasks.
    #     - Example: Basic data entry, simple scripting, or support roles.
    #     - Medium:
    #     - Moderate technical complexity or some leadership.
    #     - Example: Developed mid-sized applications, involved in system integration, led small teams.
    #     - High:
    #     - High technical complexity or significant leadership.
    #     - Example: Architected large-scale systems, led major projects, managed large technical teams.

    # 2. Domain Relevance:
    # - Identify the application domains for each experience (e.g., Insurance, Banking).
    # - Assign a domain score of [Low], [Medium], or [High] based on the following criteria:
    #     - Low:
    #     - No relevance to job description domains.
    #     - Example: Experience in unrelated fields (e.g., Retail, Manufacturing if the job is in Fintech).
    #     - Medium:
    #     - Moderate relevance to job description domains.
    #     - Example: Experience in related but not directly matching domains (e.g., General IT for a Fintech job).
    #     - High:
    #     - High relevance to job description domains.
    #     - Example: Direct experience in the same domain (e.g., Worked on Fintech projects for a Fintech job).

    # ## Output Format:

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization: 
    #     Title: 
    #     TechnicalScore: 
    #     DomainScore: 
    #     TechnicalScoreExplanation: 
    #     DomainScoreExplanation: 
    # ```

    # # Job Description

    # {}

    # # Resume Section

    # {}
    # '''.format(jd, data)

    # ##### Latest in production
    # text = \
    # '''
    # As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # # Task

    # Extract work experiences from the resume provided and present the details in YAML format, adhering to these guidelines:

    # ## Guidelines:

    # 1. Technical Rating:
    # - Assess the candidate's technical experience at each organization relative to the job description.
    # - Assign a technical score of [None], [Low], [Medium], or [High] based on the following criteria:
    #     - None: 
    #     - No relevant technical experience.
    #     - Example: Worked in a non-technical role or unrelated field.
    #     - Low:
    #     - Involvement in low-complexity technical tasks.
    #     - Example: Basic data entry, simple scripting, or support roles.
    #     - Medium:
    #     - Moderate technical complexity or some leadership.
    #     - Example: Developed mid-sized applications, involved in system integration, led small teams.
    #     - High:
    #     - High technical complexity or significant leadership.
    #     - Example: Architected large-scale systems, led major projects, managed large technical teams.

    # 2. Domain Relevance:
    # - Identify the application domains for each experience (e.g., Insurance, Banking).
    # - Assign a domain score of [Low], [Medium], or [High] based on the following criteria:
    #     - Low:
    #     - No relevance to job description domains.
    #     - Example: Experience in unrelated fields (e.g., Retail, Manufacturing if the job is in Fintech).
    #     - Medium:
    #     - Moderate relevance to job description domains.
    #     - Example: Experience in related but not directly matching domains (e.g., General IT for a Fintech job).
    #     - High:
    #     - High relevance to job description domains.
    #     - Example: Direct experience in the same domain (e.g., Worked on Fintech projects for a Fintech job).

    # ## Output Format:

    # Return the following YAML structure for each work experience entry:

    # ```yaml
    # WorkExperience:
    # - Organization: 
    #     Title: 
    #     TechnicalScore: 
    #     DomainScore: 
    # ```

    # # Job Description

    # {}

    # # Resume Section

    # {}
    # '''.format(jd, data)

    # m9/m10
    text = \
    '''
    As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # Task

    Extract work experiences from the resume provided and present the details in YAML format, adhering to these guidelines:

    ## Guidelines:

    1. Technical Rating:
    - Assess the candidate's technical experience at each organization relative to the job description.
    - Assign a technical score of [None], [Low], [Medium], or [High] based on the following criteria:
        - None: 
        - No relevant technical experience.
        - Example: Worked in a non-technical role or unrelated field.
        - Low:
        - Involvement in low-complexity technical tasks.
        - Example: Basic data entry, simple scripting, or support roles.
        - Medium:
        - Moderate technical complexity or some leadership.
        - Example: Developed mid-sized applications, involved in system integration, led small teams.
        - High:
        - High technical complexity or significant leadership.
        - Example: Architected large-scale systems, led major projects, managed large technical teams.

    2. Domain Relevance:
    - Identify the application domains for each experience (e.g., Insurance, Banking).
    - Assign a domain score of [Low], [Medium], or [High] based on the following criteria:
        - Low:
        - No relevance to job description domains.
        - Example: Experience in unrelated fields (e.g., Retail, Manufacturing if the job is in Fintech).
        - Medium:
        - Moderate relevance to job description domains.
        - Example: Experience in related but not directly matching domains (e.g., General IT for a Fintech job).
        - High:
        - High relevance to job description domains.
        - Example: Direct experience in the same domain (e.g., Worked on Fintech projects for a Fintech job).

    ## Output Format:

    Return the following YAML structure for each work experience entry:

    ```yaml
    WorkExperience:
    - Organization: 
        Title:
        DomainScore:
        TechnicalScore:
    ```

    # Job Description

    {}

    # Resume Section

    {}
    '''.format(jd, data)

    response,num_tokens = util.gemini_prompt_call(text)
    return response,num_tokens   


def parse_dates_matches(llm_raw,yoe = 0):
    college_words = ["college","university"]

    # print("YOE:",yoe,"Type YOE:",type(yoe))
    info = []
    for ele in llm_raw:
        # print("New LLM RAW Element")
        # print(ele[0],ele[1])
        dates = ele[0]
        matches = ele[1]
        # print(dates,matches)

        line_dates = [ele for ele in dates.strip().split("\n") if len(ele) > 0]
        line_matches = [ele for ele in matches.strip().split("\n") if len(ele) > 0]
        # print(line_dates,line_matches)
        
        i = 0
        while i < len(line_dates):
            # if check_organization(line_dates[max(0,i-1)]+line_dates[i]):
            if string_util.check_organization(line_dates[i]):
                # print(lines[i:i+6])
                # org = line_dates[i].strip().split(":")[-1].strip()
                # start_date = line_dates[i+1].strip().split(":")[-1].strip()
                # end_date = line_dates[i+2].strip().split(":")[-1].strip()
                # softskill_score = get_score(line_dates[i+4].strip().split(":")[1].strip().lower())

                org = line_dates[i].strip().split("Organization:")[-1].split("\n")[0].strip()
                start_date = line_dates[i+1].strip().split("StartDate:")[-1].strip()
                end_date = line_dates[i+2].strip().split("EndDate:")[-1].strip()
                # print(line_dates[i:i+5],"Current:",line_dates[i+4])
                softskill_score = get_score(line_dates[i+4].strip().split("SoftSkillsScore:")[-1].strip().lower())

                technical_score,domain_score = 0,0

                if len(start_date) > 1:
                    info.append([org,technical_score,domain_score,softskill_score,start_date,end_date])
                elif end_date.lower() == "present":
                    start_date = str(datetime.today().date())
                    info.append([org,technical_score,domain_score,softskill_score,start_date,end_date])
                else:
                    i += 1
                    continue
                for word in college_words:
                    if word in org.lower():
                        info.pop()
                        break
                
                i += 2
            i += 1
        
        # print(info)
        i = 0

        while i < len(line_matches):
            if string_util.check_organization(line_matches[max(0,i-1)]+line_matches[i]):
                # print(line_matches[max(0,i-1)]+line_matches[i])
                # org = line_matches[i].strip().split(":")[1].strip()
                # technical_score = get_score(line_matches[i+2].strip().split(":")[1].strip().lower())
                # domain_score = get_score(line_matches[i+3].strip().split(":")[1].strip().lower())


                org = line_matches[i].strip().split("Organization:")[-1].strip()
                technical_score = get_score(line_matches[i+2].strip().split("TechnicalScore:")[-1].strip().lower())
                domain_score = get_score(line_matches[i+3].strip().split("DomainScore:")[-1].strip().lower())
                # print(org,technical_score,domain_score)

                max_match,match_ele = -1,None
                for ele in info:
                    temp_ele = ele[0].replace(" ","")
                    temp_org = org.replace(" ","")
                    if fuzz.partial_ratio(temp_ele,temp_org) > max_match:
                        match_ele = ele
                        max_match = fuzz.partial_ratio(temp_ele,temp_org)

                    # if fuzz.partial_ratio(temp_ele,temp_org) > 50:
                    #     print(ele[0],org,fuzz.partial_ratio(temp_ele,temp_org),temp_ele,temp_org)
                    # if fuzz.partial_ratio(temp_ele,temp_org) > 99:
                    #     # Index Technical,Domain Score: 1,2
                    #     ele[1] = max(ele[1],technical_score)
                    #     ele[2] = max(ele[2], domain_score)
                    #     # break
                if max_match > 50:
                    # Index Technical,Domain Score: 1,2
                    # print("Updating for:",match_ele)
                    match_ele[1] = max(match_ele[1],technical_score)
                    match_ele[2] = max(match_ele[2], domain_score)
                    # print("Updated:",match_ele)
                    # break
                i += 2
            i += 1
        
    # print(info)

    # for i in range(len(info)):
    #     # print("Current Row:",info[i])
    #     if not isinstance(info[i][-1],date) and info[i][-1].lower() == "present":
    #         ans = "present"
    #     else:
    #         ans = parse_date(info[i][-1])

    #     info[i].append(parse_date(info[i][-2]))
    #     info[i].append(ans)
    
    for i in range(len(info)):
        # print(info[i-1:i+1])
        # print("Current Row:",info[i])
        if info[i][-1].lower() == "present":
            ans = "present"
        else:
            ans = util.parse_date(info[i][-1])
        
        temp_start = util.parse_date(info[i][-2])
        temp_ans = datetime.today().date() if ans == "present" else ans
        # print("Org:",info[i])
        # print("OG Estimate:",temp_start,temp_ans)

        if temp_start > temp_ans:
            # Making the above sign >= removes all unemployment gaps.
            if i == 0:
                if i + 1 < len(info):
                    if util.parse_date(info[i+1][-2]) >= temp_ans:
                        temp_start = temp_ans - relativedelta(years=2)
                    else:
                        temp_start = util.parse_date(info[i+1][-1])
                else:
                    temp_start = temp_ans - relativedelta(years=2)
            
            elif i == len(info) - 1:
                if info[i-1][-2] >= temp_ans:
                    temp_start = temp_ans - relativedelta(years=2)
                else:
                    temp_start = util.parse_date(info[i-1][-1])            
            
            else:
                # print("Previous:",info[i - 1])
                # print("Current:",info[i])
                # print("Next:",info[i + 1])

                if info[i-1][-2] >= temp_ans:
                    temp_start = util.parse_date(info[i + 1][-1])
                else:
                    temp_start = util.parse_date(info[i - 1][-1])


        #     elif i == len(info) - 1:
        #         temp_start = info[i - 1][-2]


        # elif (isinstance(ans,date) and temp_start > ans):
        #     if i == 0:
        #         if i + 1 < len(info):
        #             pass
        #         else:
        #             temp_start = datetime.today().date() - relativedelta(years=2)
        #         # Next elements start date is after currents end date
        #         if check_if and i+1 < len(info) and parse_date(info[i + 1][-2]) >= ans:
        #             temp_start = info[i - 1][-2]
        #             check_if = False


        #         # Next elements end date is less than currents end date
        #         next_end = parse_date(info[i + 1][-1])
        #         if next_end <= ans:
        #             temp_start = next_end
        #             check_if = False
        #     if check_if and i == len(info) - 1:
        #         # Before elements start date is after currents end date
        #         if check_if and i-1 > -1 and info[i-1][-2] >= ans:
        #             temp_start = parse_date(info[i + 1][-2])
        #             check_if = False
        #         # Before elements end date is less than currents end date
        #         before_end = info[i-1][-1]
        #         if before_end <= ans:
        #             temp_start = before_end
        #             check_if = False
        #     # Next elements start date is after currents end date
        #     elif parse_date(info[i + 1][-2]) >= ans:
        #         temp_start = info[i - 1][-1]
        #     # Before elements end date is less than currents end date
        #     elif info[i-1][-1] <= ans:
        #         temp_start = parse_date(info[i + 1][-2])
                
        # print("Final Answers:",temp_start,temp_ans)
        info[i].append(temp_start)
        info[i].append(ans)

    info.sort(key = lambda x : x[-2])

    for i in range(len(info)):
        if info[i][-2] == datetime.today().date():
                info[i][-2] = info[i - 1][-1]
        # print(info[i])
        if not isinstance(info[i][-1],date) and info[i][-1] == "present":
            if i == len(info) - 1:
                info[i][-1] = datetime.today().date()
            else:
                info[i][-1] = info[i + 1][-2]
        # Add extra experience for the last role. So make cur_exp > 0 condn.
        # cur_exp = diff_in_years(start_date = info[i][-2], end_date = info[i][-1])
        # if cur_exp > 0 :
        #     if info[i][-2] == datetime.today().date():
        #         info[i][-2] = info[i - 1][-1]

    # print("Before Merge")
    # print("----")
    # for ele in info:
    #     print(ele)
    # print("----")

    info = string_util.merge_experience_list(info)

    # print("Post Merge")
    # print("----")
    # for ele in info:
    #     print(ele)
    # print("----")

    total_exp,technical_exp,domain_exp,softskills_exp = 0,0,0,0
    # print(info)

    i = 0
    while i < len(info):
        # print(info[i])
        cur_exp = util.diff_in_years(start_date = info[i][-2], end_date = info[i][-1])

        year_wt = (datetime.today().date().year - info[i][-1].year)
        # print(year_wt,datetime.today().date(), info[i][-1])

        if cur_exp > 0 :
            # total_exp += cur_exp
            # technical_exp += info[i][1] * cur_exp
            # domain_exp += info[i][2] * cur_exp
            # softskills_exp += info[i][3] * cur_exp

            total_exp += cur_exp
            technical_exp += info[i][1] * cur_exp * 0.99**year_wt
            domain_exp += info[i][2] * cur_exp * 0.99**year_wt
            softskills_exp += info[i][3] * cur_exp * 0.99**year_wt


            # total_exp += cur_exp
            # technical_exp += info[i][1] * cur_exp * 0.99**(len(info) - i - 1)
            # domain_exp += info[i][2] * cur_exp * 0.99**(len(info) - i - 1)
            # softskills_exp += info[i][3] * cur_exp * 0.99**(len(info) - i - 1)

            # total_exp += cur_exp
            # technical_exp += info[i][1] * cur_exp
            # technical_wt += info[i][1]
            # domain_exp += info[i][2] * cur_exp
            # domain_wt += info[i][2]
            # softskills_exp += info[i][3] * cur_exp
            # softskills_wt += info[i][3]            

            # total_exp += cur_exp
            # technical_exp += info[i][1]
            # domain_exp += info[i][2]
            # softskills_exp += info[i][3]           
            i += 1
        else:
            info.pop(i)
    # technical_score = ((technical_exp/technical_wt)/total_exp)
    # domain_score = ((domain_exp/domain_wt)/total_exp)
    # softskill_score = ((softskills_exp/softskills_wt)/total_exp)

    # technical_score = (technical_exp/len(info))
    # domain_score = (domain_exp/len(info))
    # softskill_score = (softskills_exp/len(info))

    # technical_score = technical_exp
    # domain_score = domain_exp
    # softskill_score = softskills_exp
    # total_score = 1/(1 + np.exp(-(total_exp - yoe)))

    print("Total,Technical,Domain,Softskills:{:.3f},{:.3f},{:.3f},{:.3f}".format(total_exp,technical_exp,domain_exp,softskills_exp))
    technical_score = sigmoid(technical_exp - yoe)
    domain_score = sigmoid(domain_exp - yoe)
    softskill_score = sigmoid(softskills_exp - yoe)
    total_score = sigmoid(total_exp - yoe)
    print("Total,Technical,Domain,Softskills Score:{:.3f},{:.3f},{:.3f},{:.3f}".format(total_score,technical_score,domain_score,softskill_score))
    
    final_score = 0.25*total_score + 0.55*technical_score + 0.05*softskill_score + 0.15*domain_score


    exp_eval = {
        "exp_list" : info,

        "total_exp" : total_exp,
        "technical_exp" : technical_exp,
        "domain_exp" : domain_exp,
        "softskill_exp" : softskills_exp,

        "total_score" : round(total_score,3),
        "technical_score" : round(technical_score,3),
        "domain_score" : round(domain_score,3),
        "softskill_score" : round(softskill_score,3),
        "final_score" : round(final_score,3)
    }

    return exp_eval