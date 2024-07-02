import numpy as np
import src.util as util
import src.string_util as string_util
from datetime import datetime,date
import re
from thefuzz import fuzz
from dateutil.relativedelta import relativedelta
from typing import List, Union, Tuple

# Define scores for different levels
scores = {
    "none" : 0.05,
    "low" : 0.33,
    "medium" : 0.67,
    "high" : 1
}

def get_score(key: str) -> float:
    """
    Retrieves the score associated with a given key.

    Args:
    - key (str): Key representing the level of score (none, low, medium, high).

    Returns:
    - float: Score corresponding to the key. If key not found, returns 0.
    """
    global scores
    pattern = r'\b(none|low|medium|high)\b'
    key = re.findall(pattern, key)[0]
    key = key.strip()
    return scores.get(key, 0)

def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function for a given input.

    Args:
    - x (float): Input value.

    Returns:
    - float: Sigmoid value of x.
    """
    return 1 / (1 + np.exp(-x))

def get_baseline_score(yoe: int) -> dict:
    """
    Calculates the baseline score based on years of experience (YOE).

    Args:
    - yoe (int): Years of experience.

    Returns:
    - dict: Dictionary containing baseline information:
        - total_exp: Total years of experience.
        - technical_exp: Calculated technical experience score.
        - domain_exp: Calculated domain experience score.
        - softskill_exp: Calculated soft skill experience score.
        - total_score: Rounded total score.
        - technical_score: Rounded technical score.
        - domain_score: Rounded domain score.
        - softskill_score: Rounded soft skill score.
        - final_score: Rounded final composite score.
    """
    total_score = sigmoid(0)
    duration_weight = 0.99 ** (yoe / 2)

    technical_exp = scores["medium"] * yoe * duration_weight
    technical_score = sigmoid(technical_exp - yoe)

    domain_exp = scores["low"] * yoe * duration_weight
    domain_score = sigmoid(domain_exp - yoe)

    softskill_exp = scores["medium"] * yoe * duration_weight
    softskill_score = sigmoid(softskill_exp - yoe)

    # print("YOE,Weight,Technical Exp,Domain Exp,Softskill Exp:", yoe, duration_weight, technical_exp, domain_exp, softskill_exp)
    # print("Baseline:Total,Technical,Domain,Softskill:", total_score, technical_score, domain_score, softskill_score)

    final_score = 0.25 * total_score + 0.55 * technical_score + 0.05 * softskill_score + 0.15 * domain_score
    # print("Expected Final Baseline:", final_score)

    baseline = {
        "total_exp" : yoe,
        "technical_exp" : technical_exp,
        "domain_exp" : domain_exp,
        "softskill_exp" : softskill_exp,

        "total_score" : round(total_score, 3),
        "technical_score" : round(technical_score, 3),
        "domain_score" : round(domain_score, 3),
        "softskill_score" : round(softskill_score, 3),
        "final_score" : round(final_score, 3)
    }
    return baseline


def get_experience_gemini(jd: str, data: str) -> Tuple[str, int]:
    """
    Generate a Gemini prompt for extracting work experiences from resumes.

    Args:
    - jd (str): Job description to be included in the prompt.
    - data (str): Resume section containing work experience details.

    Returns:
    - response (str): Generated prompt response in YAML format.
    - num_tokens (int): Number of tokens used in the prompt.

    Format:
    - Constructs a prompt template with job description and resume data.
    - Calls util.gemini_prompt_call to generate response based on provided text.

    YAML Structure (for each work experience):
    WorkExperience:
    - Organization: Name of the organization.
    - StartDate: Start date of employment.
    - EndDate: End date of employment (use "present" for current positions).
    - Title: Job title held.
    - TechnicalScore: Assessment of technical skills alignment ([None], [Low], [Medium], [High]).
    - DomainScore: Assessment of domain relevance ([Low], [Medium], [High]).
    - DomainScoreExplanation: Explanation of domain score based on project domains.
    - SoftSkillsScore: Assessment of soft skills alignment ([Low], [Medium], [High]).
    """

    # Constructing the prompt text template
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
        - Identify application domains for the individual projects. (e.g., Insurance, Banking, Fintech, etc.). 
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
    '''.format(jd, data)

    # Calling util.gemini_prompt_call to generate response
    response, num_tokens = util.gemini_prompt_call(text)

    return response, num_tokens


def get_gemini_single_prompt(jd: str, data: str) -> tuple:
    """
    Generates a Gemini prompt for evaluating resume against a job description.

    Args:
    - jd (str): Job description text to be inserted into the prompt template.
    - data (str): Resume data to be inserted into the prompt template.

    Returns:
    - tuple: A tuple containing the response YAML structure and the number of tokens used.
    """
    # Template for the Gemini prompt
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
    '''.format(jd, data)

    # Call to the Gemini API or function for generating prompt
    response, num_tokens = util.gemini_prompt_call(text)

    return response, num_tokens


def get_dates_prompt(jd: str, data: str) -> tuple[str, int]:
    """
    Generates a prompt for extracting work experience details and rating soft skills based on given job description and resume data.

    Args:
    - jd (str): Job description text to be incorporated into the prompt.
    - data (str): Resume section containing work experience details to be formatted into YAML.

    Returns:
    - response (str): Prompt text formatted with job description and resume data.
    - num_tokens (int): Number of tokens in the generated prompt.

    Prompt Structure:
    - The function creates a structured prompt using formatted strings.
    - It incorporates the job description and resume data into a YAML structure for work experience details.
    - Soft skills are rated based on the job description, using [Low], [Medium], or [High] ratings.

    YAML Structure Example:
    ```yaml
    WorkExperience:
    - Organization: 
      StartDate: 
      EndDate: 
      Title: 
      SoftSkillsScore: 
    ```

    Usage Example:
    ```
    jd = "Seeking a candidate with strong communication and problem-solving skills."
    data = "Worked as a software developer from Jan 2020 to present."
    response, num_tokens = get_dates_prompt(jd, data)
    ```
    """
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

    response, num_tokens = util.gemini_prompt_call(text)
    return response, num_tokens



def get_match_prompt(jd: str, data: str) -> Tuple[str, int]:
    """
    Generates a prompt for a recruiter to assess candidate's work experiences against a job description.

    Args:
    - jd (str): The job description provided as a string.
    - data (str): The resume data to be assessed.

    Returns:
    - response (str): Generated prompt text with placeholders filled.
    - num_tokens (int): Number of tokens in the generated prompt.

    Raises:
    - Assumes `util.gemini_prompt_call(text)` is defined elsewhere and handles the generation of prompt.

    Notes:
    - This function dynamically generates a prompt tailored for recruiter assessment based on provided job description and resume data.
    - Utilizes string formatting to inject `jd` and `data` into the prompt template.
    - Assumes `util.gemini_prompt_call(text)` is a function that takes a string prompt and returns response text and token count.

    Example usage:
    response, num_tokens = get_match_prompt("Job description text here", "Resume data here")
    """
    text = \
    '''
    As an experienced recruiter with over two decades in the IT industry, you've successfully placed numerous candidates across various levels in diverse client organizations. Now, here's your task:

    # Task

    Extract work experiences from the resume provided and present the details in YAML format, adhering to these guidelines:

    ## Guidelines:

    1. Technical Rating:
    - Assess the candidate's technical experience at each organization in relation to the job description provided.
    - Assign a technical score of [None], [Low], [Medium], or [High] based on the following criteria:
        - None: 
        - Experience is not technical or no related to job description.
        - Example: Worked in a non-technical role or skills were unrelated to job description.
        - Low:
        - Low relevance to technical skills and experiences required by the job.
        - Example: Experience marginally related to the technical requirements of the job.
        - Medium:
        - Experience involving moderate technical complexity or some leadership that partially matches the technical requirements of the job description.
        - Example: Developed mid-sized applications, involved in system integration, led small teams with skills and experiences aligned with the job’s technical requirements.
        - High:
        - High technical complexity or significant leadership experience that closely matches the technical requirements of the job description.
        - Example: Architected large-scale systems, led major projects, managed large technical teams with skills and experiences highly relevant to the job’s technical requirements.

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


    # Job Description

    {}

    # Resume Section

    {}
    '''.format(jd, data)

    # Assuming util.gemini_prompt_call(text) generates response and token count
    response, num_tokens = util.gemini_prompt_call(text)
    return response, num_tokens


def parse_llmop_style1(cur_data: str) -> List[List[str]]:
    """
    Parses structured data from 'cur_data' formatted in a specific style (LLM OP Style 1).
    
    Args:
    - cur_data (str): Raw data string containing structured information.
    
    Returns:
    - List of lists, each containing:
      [organization (str), technical score (int), domain score (int), soft skill score (int), start date (str), end date (str)]
    """
    college_words = ["college", "university"]
    lines = [ele for ele in cur_data.strip().split("\n") if len(ele) > 0]
    i = 0
    info = []

    while i < len(lines):
        if string_util.check_organization(lines[max(0, i - 1)] + lines[i]):
            org = lines[i].strip().split(":")[-1].strip()
            start_date = lines[i + 1].strip().split(":")[-1].strip()
            end_date = lines[i + 2].strip().split(":")[-1].strip()

            technical_score = get_score(lines[i + 4].strip().split(":")[1].strip().lower())
            domain_score = get_score(lines[i + 5].strip().split(":")[1].strip().lower())
            softskill_score = get_score(lines[i + 6].strip().split(":")[1].strip().lower())

            if len(start_date) > 1:
                info.append([org, technical_score, domain_score, softskill_score, start_date, end_date])
            elif end_date.lower() == "present":
                start_date = str(datetime.today().date())
                info.append([org, technical_score, domain_score, softskill_score, start_date, end_date])
            else:
                i += 1
                continue
            
            # Check if the organization contains any college-related keywords
            for word in college_words:
                if word in org.lower():
                    info.pop()  # Remove the last appended item if it's a college or university
                    break
            
            i += 3  # Move to the next block of data after processing

        i += 1  # Move to the next line in the data
    
    return info



def parse_llmop_style2(llm_raw: list) -> list:
    """
    Parses structured data from `llm_raw` list and extracts relevant information about organizations,
    technical scores, domain scores, soft skills scores, start dates, and end dates.

    Args:
    - llm_raw (list): List of tuples containing structured data elements (dates, matches).

    Returns:
    - info (list): List of lists containing parsed information for each organization, including scores and dates.
    """
    college_words = ["college", "university"]
    info = []

    for ele in llm_raw:
        dates = ele[0]
        matches = ele[1]

        line_dates = [ele.strip() for ele in dates.strip().split("\n") if len(ele.strip()) > 0]
        line_matches = [ele.strip() for ele in matches.strip().split("\n") if len(ele.strip()) > 0]

        i = 0
        while i < len(line_dates):
            if string_util.check_organization(line_dates[i]):
                org = line_dates[i].strip().split("Organization:")[-1].strip()
                start_date = line_dates[i+1].strip().split("StartDate:")[-1].strip()
                end_date = line_dates[i+2].strip().split("EndDate:")[-1].strip()
                softskill_score = get_score(line_dates[i+4].strip().split("SoftSkillsScore:")[-1].strip().lower())

                technical_score, domain_score = 0, 0

                if len(start_date) > 1:
                    info.append([org, technical_score, domain_score, softskill_score, start_date, end_date])
                elif end_date.lower() == "present":
                    start_date = str(datetime.today().date())
                    info.append([org, technical_score, domain_score, softskill_score, start_date, end_date])
                else:
                    i += 1
                    continue

                # Remove entries containing college or university from info list
                for word in college_words:
                    if word in org.lower():
                        info.pop()
                        break

                i += 2
            i += 1

        i = 0
        while i < len(line_matches):
            if string_util.check_organization(line_matches[max(0, i - 1)] + line_matches[i]):
                org = line_matches[i].strip().split("Organization:")[-1].strip()
                technical_score = get_score(line_matches[i + 2].strip().split("TechnicalScore:")[-1].strip().lower())
                domain_score = get_score(line_matches[i + 3].strip().split("DomainScore:")[-1].strip().lower())

                max_match, match_ele = -1, None
                for ele in info:
                    temp_ele = ele[0].replace(" ", "")
                    temp_org = org.replace(" ", "")
                    if fuzz.partial_ratio(temp_ele, temp_org) > max_match:
                        match_ele = ele
                        max_match = fuzz.partial_ratio(temp_ele, temp_org)

                if max_match > 50:
                    match_ele[1] = max(match_ele[1], technical_score)  # Update technical score
                    match_ele[2] = max(match_ele[2], domain_score)  # Update domain score

                i += 2
            i += 1
    return info


def order_exp(info: List[List[Union[str, date]]]) -> List[List[Union[str, date]]]:
    """
    Orders a list of job experience information based on start and end dates, handling gaps and present dates.

    Args:
    - info: A list of lists where each inner list contains job information structured as [job_title, start_date, end_date].

    Returns:
    - A sorted list of job information where gaps in employment are filled appropriately.
    """

    for i in range(len(info)):
        if info[i][-1].lower() == "present":
            ans = "present"
        else:
            ans = util.parse_date(info[i][-1])
        
        temp_start = util.parse_date(info[i][-2])
        temp_ans = datetime.today().date() if ans == "present" else ans

        if temp_start > temp_ans:
            # Handling gaps in employment
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
                if info[i-1][-2] >= temp_ans:
                    temp_start = util.parse_date(info[i + 1][-1])
                else:
                    temp_start = util.parse_date(info[i - 1][-1])

        info[i].append(temp_start)  # Appending adjusted start date
        info[i].append(ans)         # Appending parsed end date or "present"

    info.sort(key=lambda x: x[-2])  # Sorting based on adjusted start dates

    # Handling "present" dates and merging adjacent job experiences
    for i in range(len(info)):
        if info[i][-2] == datetime.today().date():
            info[i][-2] = info[i - 1][-1]  # Adjusting "present" end dates
        if not isinstance(info[i][-1], date) and info[i][-1] == "present":
            if i == len(info) - 1:
                info[i][-1] = datetime.today().date()
            else:
                info[i][-1] = info[i + 1][-2]

    info = string_util.merge_experience_list(info)  # Merging adjacent job experiences
    return info



def score_exp(info, yoe=0):
    """
    Calculate scores based on experience information.

    Args:
    - info (list): List of tuples containing experience details.
                   Each tuple format: (role, technical_exp, domain_exp, softskills_exp, start_date, end_date).
                   Where:
                   - role (str): Role or position.
                   - technical_exp (float): Technical experience in years.
                   - domain_exp (float): Domain-specific experience in years.
                   - softskills_exp (float): Soft skills experience in years.
                   - start_date (datetime.date): Start date of the experience.
                   - end_date (datetime.date): End date of the experience.
    - yoe (float, optional): Years of experience to subtract from each type of experience score. Default is 0.

    Returns:
    - exp_eval (dict): Dictionary containing:
                       - exp_list: Original list of experiences.
                       - total_exp: Total cumulative experience in years.
                       - technical_exp: Weighted technical experience score.
                       - domain_exp: Weighted domain experience score.
                       - softskill_exp: Weighted soft skills experience score.
                       - total_score: Sigmoid-transformed total experience score.
                       - technical_score: Sigmoid-transformed technical experience score.
                       - domain_score: Sigmoid-transformed domain experience score.
                       - softskill_score: Sigmoid-transformed soft skills experience score.
                       - final_score: Final combined score based on weighted averages of the above scores.
    """

    total_exp, technical_exp, domain_exp, softskills_exp = 0, 0, 0, 0

    i = 0
    while i < len(info):
        cur_exp = util.diff_in_years(start_date=info[i][-2], end_date=info[i][-1])
        year_wt = (datetime.today().date().year - info[i][-1].year)

        if cur_exp > 0:
            total_exp += cur_exp
            technical_exp += info[i][1] * cur_exp * 0.99 ** year_wt
            domain_exp += info[i][2] * cur_exp * 0.99 ** year_wt
            softskills_exp += info[i][3] * cur_exp * 0.99 ** year_wt

            i += 1
        else:
            info.pop(i)

    technical_score = sigmoid(technical_exp - yoe)
    domain_score = sigmoid(domain_exp - yoe)
    softskill_score = sigmoid(softskills_exp - yoe)
    total_score = sigmoid(total_exp - yoe)

    final_score = 0.25 * total_score + 0.55 * technical_score + 0.05 * softskill_score + 0.15 * domain_score

    exp_eval = {
        "exp_list": info,
        "total_exp": total_exp,
        "technical_exp": technical_exp,
        "domain_exp": domain_exp,
        "softskill_exp": softskills_exp,
        "total_score": round(total_score, 3),
        "technical_score": round(technical_score, 3),
        "domain_score": round(domain_score, 3),
        "softskill_score": round(softskill_score, 3),
        "final_score": round(final_score, 3)
    }

    return exp_eval