import re
from langchain.schema.document import Document
from typing import List, Tuple


# Function to check if a line contains the keyword 'Organization'
def check_organization(line: str) -> bool:
    """
    Checks if a line contains the keyword 'Organization'.

    Args:
    - line (str): The line of text to check.

    Returns:
    - bool: True if the line contains 'Organization', False otherwise.
    """
    pattern = r'-\s*\n*\t*\s*Organization\b'
    match = re.search(pattern, line)
    return bool(match)

# Function to check if a line contains a specific word
def check_word(word: str, line: str) -> bool:
    """
    Checks if a line contains a specific word.

    Args:
    - word (str): The word to search for.
    - line (str): The line of text to check.

    Returns:
    - bool: True if the word is found in the line, False otherwise.
    """
    pattern = r'-\s*\n*\t*\s*{}\b'.format(word)
    match = re.search(pattern, line)
    return bool(match)

# Function to create a list of Document objects from a string
def get_doc_frm_st(string: str = "", source: str = "") -> List[Document]:
    """
    Creates a list of Document objects from a string.

    Args:
    - string (str): The content of the document.
    - source (str): The source metadata for the document.

    Returns:
    - List[Document]: A list containing a single Document object.
    """
    return [Document(page_content=string, metadata={"source": "{}".format(source)})]

# Function to find the maximum overlap between two strings
def find_overlap(a: str, b: str) -> Tuple[int, str]:
    """
    Finds the maximum overlap between the end of string a and the start of string b.

    Args:
    - a (str): First string.
    - b (str): Second string.

    Returns:
    - Tuple[int, str]: Maximum overlap length and the merged string.
    """
    max_overlap = 0
    overlap_str = ""
    
    for i in range(1, min(len(a), len(b)) + 1):
        if a[-i:] == b[:i]:
            max_overlap = i
            overlap_str = a + b[i:]
    
    return max_overlap, overlap_str

# Function to merge a list of overlapping strings into one string
def merge_overlapping_strings(strings: List[str]) -> str:
    """
    Merges a list of overlapping strings into one string.

    Args:
    - strings (List[str]): List of strings to merge.

    Returns:
    - str: Merged string.
    """
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

# Function to merge a list of sorted experiences based on their start and end dates
def merge_experience_list(experiences_sorted: List[List]) -> Tuple[List[List], List[List]]:
    """
    Merges a list of experiences sorted by start date based on their overlapping organizations.

    Args:
    - experiences_sorted (List[List]): List of experiences, each represented as a list with organization name,
      technical experience, domain score, start date, and end date.

    Returns:
    - Tuple[List[List], List[List]]: A tuple containing two lists:
      1. Final merged list of experiences.
      2. List of experiences that could not be merged due to overlapping dates.
    """
    final_list = []
    
    for experience in experiences_sorted:
        org_name, start_date, end_date = experience[0], experience[-2], experience[-1]
        
        if not final_list:
            final_list.append(experience)
        else:
            last_experience = final_list[-1]
            last_end_date = last_experience[-1]
            
            if start_date >= last_end_date:
                final_list.append(experience)
            else:
                # Merge overlapping organization names
                merged_org_name = merge_overlapping_strings(sorted([last_experience[0], org_name], key=lambda x: len(x)))
                last_experience[0] = merged_org_name

                # Update other attributes if needed (assuming these indexes correspond to technical experience and domain score)
                last_experience[1] = max(last_experience[1], experience[1])
                last_experience[2] = max(last_experience[2], experience[2])
                last_experience[3] = max(last_experience[3], experience[3])

                # Update end date if the new experience's end date is later
                if end_date > last_end_date:
                    last_experience[-1] = end_date
                    
    return final_list