# https://stackoverflow.com/questions/16114391/adding-directory-to-sys-path-pythonpath
import sys
import os

remove_path = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.dirname(remove_path)

if remove_path in sys.path:
    sys.path.remove(remove_path)
if new_path not in sys.path:
    sys.path.insert(0,new_path)


## Imports
from src.experience import get_dates_prompt,get_match_prompt
from src.skills_match import get_jd_domain_yoe,get_jd_domain_reqs_stop_words

#### JD Prompt
def get_jd_domain_reqs(data, func_type = ""):
    if func_type == "prompt":
        return get_jd_domain_yoe(data)
    elif func_type == "stop_words":
        return get_jd_domain_reqs_stop_words(data)
    else:
        return data,""


#### Style 2 Prompts
def split_exp_score_prompt(jd_summary, resume_content):
    '''
    In this style prompt we prompt the llm twice. In the first request we get the experience YAML. In the second request we get match scores from the LLM.
    This style of prompting is defined as style 2 in this system.
    '''
    output = []
    llm_raw = []
    
    for i in range(0,len(resume_content),100000):
        cur = resume_content[max(i - 100,0):min(i+100000,len(resume_content))]
        dates = get_dates_prompt(jd_summary,cur)
        matches = get_match_prompt(jd_summary,cur)
        llm_raw.append((dates,matches))
        ans = "".join(llm_raw[-1][0][0].text + llm_raw[-1][1][0].text)
        output.append(ans)
    
    return llm_raw,output