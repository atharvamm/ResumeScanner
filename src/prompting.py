import sys
import os
from typing import List, Tuple, Any

# Remove the current directory from sys.path and add its parent directory
remove_path = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.dirname(remove_path)

if remove_path in sys.path:
    sys.path.remove(remove_path)
if new_path not in sys.path:
    sys.path.insert(0, new_path)

# Imports
import src.experience as experience
import src.skills_match as skills_match

def get_jd_domain_reqs(data: Any, func_type: str = "") -> Tuple[Any, str]:
    """
    Extract domain requirements from job description data.

    :param data: Job description data
    :param func_type: Type of function to apply ("prompt" or "stop_words")
    :return: Processed data and an additional string (depends on func_type)
    """
    if func_type == "prompt":
        return skills_match.get_jd_domain_yoe(data)
    elif func_type == "stop_words":
        return skills_match.get_jd_domain_reqs_stop_words(data)
    else:
        return data, ""

def singleprompt(jd_summary: str, resume_content: str) -> Tuple[List[Any], List[str]]:
    """
    Generate a single prompt to the LLM and get all mathematical and score details.

    :param jd_summary: Job description summary
    :param resume_content: Resume content
    :return: A tuple containing raw LLM output and processed output
    """
    output = []
    llm_raw = []
    for i in range(0, len(resume_content), 100000):
        cur = resume_content[max(i - 100, 0):min(i + 100000, len(resume_content))]
        temp = experience.get_gemini_single_prompt(jd_summary, cur)
        llm_raw.append(temp)
        ans = "".join(llm_raw[-1][0].text)
        output.append(ans)
    return llm_raw, output

def single_cot(jd_summary: str, resume_content: str) -> Tuple[List[Any], List[str]]:
    """
    Generate a single chain-of-thought prompt to the LLM for experience calculation.

    :param jd_summary: Job description summary
    :param resume_content: Resume content
    :return: A tuple containing raw LLM output and processed output
    """
    output = []
    llm_raw = []
    
    for i in range(0, len(resume_content), 100000):
        cur = resume_content[max(i - 100, 0):min(i + 100000, len(resume_content))]
        temp = experience.get_experience_gemini(jd_summary, cur)
        llm_raw.append(temp)
        ans = "".join(llm_raw[-1][0].text)
        output.append(ans)
    return llm_raw, output

def split_exp_score_prompt(jd_summary: str, resume_content: str) -> Tuple[List[Tuple[Any, Any]], List[str]]:
    """
    Generate prompts to the LLM in two stages: first for experience YAML, second for match scores.

    :param jd_summary: Job description summary
    :param resume_content: Resume content
    :return: A tuple containing raw LLM output and processed output
    """
    output = []
    llm_raw = []
    
    for i in range(0, len(resume_content), 100000):
        cur = resume_content[max(i - 100, 0):min(i + 100000, len(resume_content))]
        dates = experience.get_dates_prompt(jd_summary, cur)
        matches = experience.get_match_prompt(jd_summary, cur)
        llm_raw.append((dates, matches))
        ans = "".join(llm_raw[-1][0][0].text + llm_raw[-1][1][0].text)
        output.append(ans)
    
    return llm_raw, output
