import re
from langchain.schema.document import Document
from typing import List, Tuple

def check_organization(line):
    pattern = r'-\s*\n*\t*\s*Organization\b'
    match = re.search(pattern, line)
    return bool(match)

def check_word(word,line):
    pattern = r'-\s*\n*\t*\s*{}\b'.format(word)
    match = re.search(pattern, line)
    return bool(match)

def get_doc_frm_st(string = "", source = "" ):
    return [Document(page_content = string, metadata = {"source": "{}".format(source)})]

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

def merge_experience_list(experiences_sorted: List[List]) -> Tuple[List[List], List[List]]:

    final_list = []
    # scrap_list = []
    
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
                merged_org_name = merge_overlapping_strings(sorted([last_experience[0], org_name],key = lambda x : len(x)))
                last_experience[0] = merged_org_name

                # Technical Experience,Domain Score = 1,2
                last_experience[1] = max(last_experience[1],experience[1])
                last_experience[2] = max(last_experience[2],experience[2])
                last_experience[3] = max(last_experience[3],experience[3])

                if end_date > last_end_date:
                    last_experience[-1] = end_date
                    pass

                # if end_date <= last_end_date:
                    # merged_org_name = merge_overlapping_strings(sorted([last_experience[0], org_name],key = lambda x : len(x)))
                    # last_experience[0] = merged_org_name
                # else:

                    # scrap_list.append(experience)
                    
    return final_list