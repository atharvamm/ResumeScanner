
# https://stackoverflow.com/questions/16114391/adding-directory-to-sys-path-pythonpath
import sys
import os

remove_path = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.dirname(remove_path)
file_path = os.path.join(remove_path,"pickles")
sys.path.remove(remove_path)
sys.path.insert(0,new_path)


from src.util import load_pickle
from datetime import datetime
# https://stackoverflow.com/questions/3316882/how-do-i-get-a-string-format-of-the-current-date-time-in-python
print("Run:",datetime.now().strftime("%I:%M:%S%p on %B %d, %Y"))

import glob

def check_status(root_path,fname):
    files = glob.glob("{}/**/{}".format(root_path,fname),recursive=True)
    file_name = "".join(files).lower()
    if "final" in file_name:
        return "Final Selected"
    elif "selected" in file_name:
        return "Selected"
    else:
        return "Rejected"

def print_check_results(new_path,jd_source,check_results):
    print("Current Table:{}".format(jd_source.split("/")[-1]),end="\n"*2)
    print("-"*75)
    for ele in sorted(check_results,key = lambda x : (x[1],x[0]), reverse=True):
        fname,score,expectation = ele[0].split("/")[-1],ele[1],ele[2]
        status = check_status(new_path,fname)
        print("|","{:<30}".format(fname[:30]),"||","{:.3f}".format(score),"||","{:<10}".format(status[:10]),"||","{:<12}".format(expectation[:12]),"|")
        print("-"*75)
    return None

import argparse

parser = argparse.ArgumentParser(description="Print for resume scanner.")
parser.add_argument("--filename", required=True, help="The name of the file to process")
args = parser.parse_args()
# print(f"Processing file: {args.filename}")


data = load_pickle(os.path.join(file_path,f"{args.filename}"))

check_results = []

styles =["single_prompt","single_cot","breakexpandmatch"]
style = styles[2]


for jd_source in data.keys():
    print("JD Source:", jd_source)
    yoe = data[jd_source]["yoe"]
    check_results.clear()
    for doc_source in data[jd_source]["docs"].keys():
        print("Scanning:",doc_source)
        
        if data[jd_source]["docs"][doc_source].get("text_it","og") == "alt":
            exp_eval = data[jd_source]["docs"][doc_source]["alt_exp_eval"]
            llm_text = data[jd_source]["docs"][doc_source]["alt_llm_text"]
        else:
            exp_eval = data[jd_source]["docs"][doc_source]["exp_eval"]
            llm_text = data[jd_source]["docs"][doc_source]["llm_text"]

        print(llm_text)

        
        if style == styles[0]:
            print("Parsing")
        
        elif style == styles[2]:

            exp_list = exp_eval["exp_list"]

            total_exp = exp_eval["total_exp"]
            technical_exp = exp_eval["technical_exp"]
            domain_exp = exp_eval["domain_exp"]
            softskill_exp = exp_eval["softskill_exp"]

            total_score = exp_eval["total_score"]
            technical_score = exp_eval["technical_score"]
            domain_score = exp_eval["domain_score"]
            softskill_score = exp_eval["softskill_score"]
            final_score = data[jd_source]["docs"][doc_source]["final_score"]

            print("-"*15)
            print("-"*15)

            for exp in exp_list:
                print("- ",exp[0],"||",exp[-2].strftime("%B %Y"),"-",exp[-1].strftime("%B %Y"),"|| Tech.:",exp[1],"|| Dom:",exp[2])


            recommendation = data[jd_source]["docs"][doc_source]["recommendation"]

            print("-"*24)
            print("|| Score Comaprison ||")
            print("-"*24)
            print("Baseline:",str(data[jd_source]["baseline"]))
            print("="*10)
            temp = exp_eval
            del temp["exp_list"]
            print("Exp:",str(temp))
            check_results.append([doc_source,final_score,recommendation])

        print("\n"*2)
    print_check_results(new_path,jd_source,check_results)


