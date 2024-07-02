import pickle
import google.generativeai as genai
import os
import time
from dateutil.parser import parse
from datetime import datetime,date
from typing import Any, List, Tuple, Dict

def save_pickle(obj: Any, path: str) -> None:
    """
    Save an object to a file using pickle.

    Args:
        obj (Any): The object to be saved.
        path (str): The path where the object should be saved.
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(path: str) -> Any:
    """
    Load an object from a pickle file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.
    """
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def is_within_percentage(score: float, baseline: float, percentage: float = 5) -> bool:
    """
    Check if a score is within a certain percentage of a baseline value.

    Args:
        score (float): The score to be checked.
        baseline (float): The baseline value to compare against.
        percentage (float): The allowed percentage deviation. Default is 5%.

    Returns:
        bool: True if the score is within the allowed percentage deviation, False otherwise.
    """
    allowed_deviation = baseline * (percentage / 100.0)
    lower_bound = baseline - allowed_deviation
    upper_bound = baseline + allowed_deviation
    
    return lower_bound <= score <= upper_bound

def get_lengths(text: str) -> List[int]:
    """
    Get the lengths of segments of a text, each segment having a maximum length of 5000 characters.

    Args:
        text (str): The input text.

    Returns:
        List[int]: A list of lengths of the segments.
    """
    lengths = []
    for i in range(0, len(text), 5000):
        # Create a segment with 5000 characters, extending 100 characters back to avoid breaking words.
        cur = text[max(i - 100, 0):min(i + 5000, len(text))]
        lengths.append(len(cur))
    return lengths


def gemini_prompt_call(
    text: str, 
    model_name: str = None, 
    gen_config: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Calls a generative model to generate content based on the provided text prompt.

    Args:
        text (str): The input text prompt for the generative model.
        model_name (str, optional): The name of the model to use. Defaults to None.
        gen_config (Dict[str, Any], optional): The generation configuration. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], int]: A tuple containing the response from the model and the number of tokens in the input text.
    """
    
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]
    
    if gen_config is None:
        gen_config = {
            "temperature": 0.0,
            "top_p": 0.00,
            "top_k": 1
        }

    # Configure the genai API with the API key from the environment variable
    genai.configure(api_key=os.environ["GEMINI"])
    
    # Initialize the generative model with the specified model name
    model = genai.GenerativeModel(model_name)
    
    # Count the number of tokens in the input text
    num_tokens = model.count_tokens(text)

    success = False
    count = 0  
    
    while not success:
        try:
            count += 1
            # Generate content using the model
            response = model.generate_content(text, generation_config=gen_config)
            success = True  # If no exception occurs, set success to True
        except Exception as e:
            print(e)
            response = {"text": str(e)}
            if count == 15:
                break
            # Wait for a short time before retrying
            time.sleep(int(os.getenv("SHORT_TIME")) // 2)
        # Wait for a short time before retrying
        time.sleep(int(os.getenv("SHORT_TIME")))
    
    return response, num_tokens



def parse_date(string: str, fuzzy: bool = True) -> date:
    """
    Parses a date string and returns a date object.
    
    Parameters:
    - string (str): The date string to parse.
    - fuzzy (bool): Whether to ignore unknown tokens in the date string.
    
    Returns:
    - date: Parsed date object. If parsing fails, returns today's date.
    """
    # If the input is already a date object, return it
    if isinstance(string, date):
        return string

    # Replace common incorrect apostrophe characters with a space and '20'
    string = string.replace("'", " 20")
    string = string.replace("’", " 20")

    try:
        # Handle date strings with slashes, assuming MM/YYYY or MM/YY format
        if "/" in string:
            parts = string.split("/")
            if len(parts) in [2, 3] and len(parts[-1]) in [2, 4]:
                month = int(parts[-2])
                year = int(parts[-1]) if len(parts[-1]) == 4 else int(parts[-1]) + 2000
                cur_date = datetime(year, month, 1).date()
                return cur_date

        # Use dateutil.parser to handle other date formats
        cur_date = parse(string, fuzzy=fuzzy).date()
        return cur_date

    except Exception as e:
        # Handle OverflowError and ValueError specifically
        if isinstance(e, (OverflowError, ValueError)):
            print(f"{e} - Could Not Parse")
        # Return today's date if parsing fails
        return datetime.today().date()


def diff_in_years(start_date: date, end_date: date) -> float:
    """
    Calculates the difference in years between two dates.
    
    Parameters:
    - start_date (date): The start date.
    - end_date (date): The end date.
    
    Returns:
    - float: The difference in years.
    """
    # Calculate the difference in years
    year_diff = end_date.year - start_date.year
    # Calculate the difference in months
    month_diff = end_date.month - start_date.month
    
    # Total number of months between the two dates
    total_months = year_diff * 12 + month_diff
    
    # Convert total months to years
    total_years = total_months / 12
    return total_years