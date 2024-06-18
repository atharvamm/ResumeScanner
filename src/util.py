import pickle
import re
import google.generativeai as genai
import os
import time
from dateutil.parser import parse
from datetime import datetime,date

def save_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def is_within_percentage(score, baseline, percentage = 2):

    allowed_deviation = baseline * (percentage / 100.0)    
    lower_bound = baseline - allowed_deviation
    upper_bound = baseline + allowed_deviation
    
    return lower_bound <= score <= upper_bound

def get_lengths(text):
    lengths = []
    for i in range(0,len(text),5000):
        cur = text[max(i - 100,0):min(i+5000,len(text))]
        # lengths.append([cur,len(cur)])
        lengths.append(len(cur))
    return lengths


def gemini_prompt_call(text,model_name = None,gen_config = None):
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]
    if gen_config is None:
            gen_config = {
            "temperature": 0.0,
            "top_p": 0.00,
            "top_k": 1
        }

    genai.configure(api_key=os.environ["GEMINI"])
    model = genai.GenerativeModel(model_name)
    num_tokens = model.count_tokens(text)
    # info = genai.get_model("models/"+model_name)
    # print((info.input_token_limit, info.output_token_limit))

    success = False  
    while not success:
        try:
            response = model.generate_content(text,generation_config=gen_config)
            success = True  # If no exception occurs, set success to True
        except Exception as e:
            print(e)
            response = {"text": str(e)} 
            time.sleep(int(os.getenv("SHORT_TIME"))//2)
        time.sleep(int(os.getenv("SHORT_TIME")))
    return response,num_tokens

def parse_date(string, fuzzy=True):
    if isinstance(string,date):
        return string
    # print(string,type(string), end = ">><<\n")
    string = string.replace("'"," 20")
    string = string.replace("’"," 20")
    # string = string.translate({"'":"20","’":"20"})
    # print(string)

    try: 
        if "/" in string:
            parts = string.split("/")
            if len(parts) in [2,3] and len(parts[-1]) in [2,4]:
                month = int(parts[-2])
                year = int(parts[-1]) if len(parts[-1]) == 4 else int(parts[-1]) + 2000
                cur_date = datetime(year, month, 1).date()
                return cur_date

        cur_date = parse(string, fuzzy=fuzzy).date()
        # print("Returning:",cur_date)
        return cur_date
        # return "{}".format(cur_date.strftime("%B %Y"))

    except Exception as e:
        # if e in [OverflowError,ValueError,ParserError]:
        # return f"{e} - Could Not Parse"
        return datetime.today().date()
    
def diff_in_years(start_date: datetime, end_date: datetime) -> int:

    year_diff = end_date.year - start_date.year
    month_diff = end_date.month - start_date.month
    
    total_months = year_diff * 12 + month_diff
    
    total_years = total_months / 12
    return total_years