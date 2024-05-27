
# 1. Code to merge intervals
# 2. Code to find total experience
# 3. Code to find gaps in experience
# 4. Code to find experience start and end as list from resume.
from openai import OpenAI
import os
from src.process_text import process_text
from src.parse_doc import text_splitter
from src.vector_store import get_vectorstore
from nltk.corpus import stopwords

def experience_all(data):
    import re
    import datetime
    
    regex_strings = [
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:'|\s*)(?:\d{4}|\d{2})",
    ]

    data = data.split(" ")
    for i,ele in enumerate(data):
        if "-" in ele:
            x = " ".join(data[i - 5: i + 5]).lower()
            print("\n",x,end = " ")
            for regex_string in regex_strings:
                results = re.findall(regex_string,x)
                if ("present".lower() in x or "till date".lower() in x) and len(results) == 1:
                    results.append(datetime.datetime.now().date())
                if len(results) > 0:
                    print(results)
                if len(results) > 1:
                    break


import re
date_pattern = re.compile(
    r"""
    (?:
        \b(?:0?[1-9]|[12][0-9]|3[01])         # Match day (01-31)
        ([-/\s.])                             # Separator
        (?:0?[1-9]|1[0-2])                    # Match month (01-12)
        \1                                    # Repeat the separator
        (?:\d{2,4})\b                         # Match year (yy or yyyy)
    |
        \b(?:0?[1-9]|[12][0-9]|3[01])         # Match day (01-31)
        ([-/\s.'])                            # Separator
        (?:\d{2})\b                           # Match year (yy)
    |
        \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)  # Match abbreviated month names
        (?:uary|ruary|ch|il|e|y|e|ust|ober|ember)?     # Match full month names
        \s?`?'?\d{2,4}                         # Match year with optional apostrophe (yy or yyyy)
        (?:\b|[\s-])
    |
        \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)  # Match abbreviated month names
        \s?'?\d{2,4}\b                         # Match year with optional apostrophe (yy or yyyy)
    |
        \b\d{4}\s*[-\s]\s*\d{4}\b
    )
    """, re.VERBOSE | re.IGNORECASE)

def contains_date(text):
    return bool(date_pattern.search(text))


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


from dateutil.parser import parse
from dateutil.parser._parser import ParserError

def is_date(string, fuzzy=True):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except Exception as e:
        if e in [OverflowError,ValueError,ParserError]:
            return False
        # else:
            # print(e)

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def clean_exp_llm(data):
    lines = data.split("\n")
    strings = []
    for i in range(len(lines)):

        start,end = 0,len(lines[i])
        while start < end:
            last_index = min(start + 150,end)
            cur_str = lines[i][start:last_index]
            # print("Cur String: ",cur_str)
            if contains_date(cur_str):
                # print("Adding: ", cur_str)
                strings.append(cur_str)
            elif has_numbers(cur_str[-25:]):
                cur_str = lines[i][start:min(last_index + 51,end)]
                # print("Cur String: ",cur_str)
                if contains_date(cur_str):
                    # print("Adding: ", cur_str)
                    strings.append(cur_str)
            start += 100

    bef,aft = lines[max(i-2,0):i],lines[i+1:min(i+3,len(lines) - 1)]

    # if "," in bef:
    #     bef =  bef.split(",")[-1]
    # if "," in aft:
    #     aft =  aft.split(",")[0]

    # print("Before:",bef)
    # print("After:",aft)
    strings = bef + strings
    strings.extend(aft)
    # strings.append("\n\n")

    string = merge_overlapping_strings(strings)
    return string


def experience_llm(text):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "{}".format(os.getenv("NVIDIA_KEY"))
    )
    
    # text = "{}\n\n Extract the beginning, end, and organization of experiences from the resume and provide them as a list of tuples (Begin,End, Org). Output only the list and nothing else.".format(text)

    text = '''
    Below is my resume. Please analyze it and provide the following information:
    1. Total Experience: Calculate the total number of years and months of work experience.
    2. Employment Gaps: Identify any significant employment gaps (gaps of more than 3 months) and provide the duration of each gap.
    3. Average Tenure: Calculate the average tenure of my jobs in years and months.
    4. Explanation: Explain how you calculated the total experience and identified the employment gaps.
    \n\n
    Resume:
    {}

    Format your response as follows:
    Total Experience: <Answer>,
    Employment Gaps: <Answer>,
    Average Tenure: <Answer>,
    Explanation: <Answer>
    '''.format(text)

    completion = client.chat.completions.create(
    model="meta/llama2-70b",
    messages=[{"role":"user","content":"{}".format(text)}],
    temperature=0.2,
    top_p=1,
    max_tokens=1024,
    stream=True
    )
    output = []
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            output.append(chunk.choices[0].delta.content)

    return output
    





def experience_rag(data):
    data.page_content = process_text(data.page_content)
    texts = text_splitter(data)

    vectorstore = get_vectorstore("resume_coll",new_conn = True)
    vectorstore.add_documents(texts)
    results = vectorstore.similarity_search_with_score("What are the start and end date of experiences ? ", k = 5)




    pass


'''
# get_connection()
# vectorstore = get_vectorstore("resume_coll",new_conn = True)
# vectorstore.add_documents(docs_data)

vectorstore = get_vectorstore("resume_coll",new_conn = False)

results = vectorstore.similarity_search_with_score(job_description, k = 5)

# results = vectorstore.max_marginal_relevance_search_with_score(job_description,k = 20, fetch_k=20,lambda_mult=0)

for result in results:
    # print(result)
    print(result[0].metadata["source"])
    # print(result)
    # print("\n"*2)

# with open(os.path.join(get_root_path(),"dump","embedding.txt"),"w") as f:
#     f.write(str(get_embedding(job_description)[0]))
# print("Final!!!")


## Temp

# Score = print(vectorstore.similarity_search_with_score("machine learning", k = 3)[0][1])

'''



if __name__ == "__main__":
    with open("/Users/atharvamhaskar/Documents/ResumeScanner/text.txt","r") as f:
        data = f.read()
    experience_all(data)

'''

# ----

# 1. Pattern 1 : String followed by number
import re

text = "Write regex pattern to match: Jan'20,Jan '20, Jan '20,January '20, March '20, Jan 2020, November2020"

patterns = re.findall(r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*(?:'|\s*)(?:\d{4}|\d{2})", text)

print(patterns)


# 



'''


# - Tried pdf plumber to pull date strings
# -




# ['', ' ', " (Sep '22, Present, Shopse)\n(Jun '20, Jul '22, Ascent Cyber Sol", "utions)\n(Jun '19, Jun '20, Avataar.ai)\n(Jul '19, Jun '20", ", Quantiphi)\n(Jul '19, Jun '20, Embibe - Ascent Cyber Solutions)\n(Jul '", "19, Jun '20, Trace VFX[Technicolor])\n(Sep '18, May '19, Mob", 'ily', 'te[Infostride])', '']


# ['', ' ', ' (MARCH 2024, PRESENT, STRATASYS)\n(FEBRUARY 2022, MAR', 'CH 2024, INFOGAIN PVT LTD)\n(DECEMBER 2020, JANUARY', ' 2022, PERSISTENT SYSTEM LTD)\n(APRIL 2017, DECEMBER', ' ', '2020, SENSEGIZ TECHNOLOGY)', '']

