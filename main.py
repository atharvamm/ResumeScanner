# from src.load_env import load_env_file, get_root_path
# from src.parse_doc import parse_dir
# from src.vector_store import get_vectorstore,get_connection,get_embedding
# from src.process_text import process_text
# from src.experience import *
# from src.misc import misc_prompt
# from src.skills_match import *
# import os
# import time
# from util import save_pickle,load_pickle


# load_env_file(".env")

# root_path = get_root_path()
# projects_path = os.path.join(root_path,"docs","projects")
# count = 0
# for project in os.listdir(projects_path):
#     if "DS_Store" in project:
#         continue

#     project_path = os.path.join(projects_path,project)
#     jd_path = os.path.join(projects_path,project,"JD")
#     resumes_path = os.path.join(projects_path,project,"resumes")

#     # # # # Read Files
#     # jd_data = parse_dir(jd_path)
#     # resume_data = parse_dir(resumes_path)   

#     # # # Save Pickles
#     # save_pickle(jd_data,os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
#     # save_pickle(resume_data,os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

#     # Load Pickles
#     jd_data = load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
#     resume_data = load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

#     # print([data.metadata["source"] for data in jd_data])
#     # print([data.metadata["source"] for data in resume_data])
#     # print("\n"*2)

#     recommendations = {}
#     for jd_doc in jd_data:

#         jd_key = jd_doc.metadata["source"]
#         recommendations[jd_key] = {}
#         jd_skills = pick_set(jd_doc.page_content.lower())
#         jd_skills_keypoints = jd_prompt(jd_skills)
#         recommendations[jd_key]["jd_skills_keypoints"] = jd_skills_keypoints
#         print(jd_key)
#         print(jd_skills_keypoints)
#         # break
#         time.sleep(20)

#         recommendations[jd_key]["docs"] = {}
#         for resume_doc in resume_data:

#             text_key = resume_doc.metadata["source"]
#             resume_doc = " ".join(process_text(resume_doc.page_content))


#             ## Run for Algorithm Skills
#             # resume_skills = pick_set(data)
#             # ans = rag_novectorstore_skills(query=jd_skills,text = resume_skills)
#             # print(ans)
#             ## Run for Algorithm Skills


#             # # ## Run for LLM Skills
#             # # resume_skills = pick_set(data)
#             # # print(data,len(data))
#             # recommendations[jd_key]["docs"][text_key] = []
#             # for i in range(0,len(resume_doc),3000):
#             #     cur = resume_doc[max(i - 100,0):min(i+3000,len(resume_doc))]
#             #     ans = "".join(skills_prompt(jd_skills_keypoints,cur))
#             #     recommendations[jd_key]["docs"][text_key].append(ans)
#             #     time.sleep(30)
#             # print(jd_key,text_key)
#             # print(jd_skills_keypoints)
#             # print("Final:",recommendations[jd_key]["docs"][text_key])
#             # # ## Run for LLM Skills


#             # ## Run for Algorithm Experience
#             # ans = clean_exp(data)
#             # print(experience_algo(ans))
#             # ## Run for Algorithm Experience


#             # ## Run for LLM Experience
#             # ans = clean_exp_llm(data)
#             # print(ans)
#             # print(experience_llm(ans))
#             # ## Run for LLM Experience


#             # ## Run for Misc
#             # print(data,len(data))
#             # prev = " "
#             # for i in range(0,len(data),2500):
#             #     cur = data[max(i - 100,0):min(i+2500,len(data))]
#             #     prev = "".join(misc_prompt(prev,cur))
#             #     # print("Prev:",prev)
#             #     # print("Cur:",cur)
#             #     time.sleep(45)
#             # print("Final:",prev)
#             # ## Run for Misc


#             # print(ans,len(ans))
#             # print(data.split("\n"))
#             # data = process_resume(data.page_content)
#             # import datefinder
#             # matches = datefinder.find_dates(data)
#             # for match in matches:
#             #     print(match)
#             # experience_all(data)
#             print("\n"*4)
#             time.sleep(60)
#     #         break
#     #     break
#     # break
# save_pickle(recommendations,os.path.join(root_path, "dump","pickles","recommendations.pkl"))


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


'''
##################
Latest Attempt
##################
'''
import sys
import os
import time

import numpy as np


from src.parse_doc import text_splitter
from src.vector_store import get_embedding
from src.load_env import load_env_file, get_root_path
from src.parse_doc import parse_dir
import src.util as util
from src.process_text import process_text
from src.experience import scan_all_experience,get_experience_gemini,get_gemini_single_prompt, get_baseline_score,parse_dates_matches
import src.skills_match as skills_match
import src.prompting as prompting


# from src.vector_store import get_vectorstore,get_connection,get_embedding

# from src.experience import *
# from src.misc import misc_prompt
# from src.skills_match import *
# import os
# import time




if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print for resume scanner.")
    parser.add_argument("--filename", required=True, help="The name of the file to process")
    args = parser.parse_args()
    print(f"Running: {args.filename}")



    load_env_file(".env")
    root_path = get_root_path()
    projects_path = os.path.join(root_path,"docs","projects")
    count = 0
    recommendations = {}
    
    styles =["single_cot","allinexp","breakexpandmatch"]
    style = styles[2]
    
    for project in os.listdir(projects_path):
        if "DS_Store" in project:
            continue
        print("Current Project: ",project)

        project_path = os.path.join(projects_path,project)
        jd_path = os.path.join(projects_path,project,"JD")
        resumes_path = os.path.join(projects_path,project,"resumes")

        # # # # Read Files
        # jd_data = parse_dir(jd_path)
        # resume_data = parse_dir(resumes_path)   

        # # # Save Pickles
        # save_pickle(jd_data,os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
        # save_pickle(resume_data,os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

        # Load Pickles
        jd_data = util.load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
        resume_data = util.load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

        # print([data.metadata["source"] for data in jd_data])
        # print([data.metadata["source"] for data in resume_data])
        # print("\n"*2)

        # print(jd_data)
        # print(resume_data)

        jd_data = jd_data[0]
        jd_key = jd_data.metadata["source"]

        # Get job description summary
        # requirement,domain = get_jd_domain_reqs(jd_data.page_content,func_type = "prompt")
        domain,yoe =  prompting.get_jd_domain_reqs(jd_data.page_content,func_type = "prompt")
        jd_summary = jd_data.page_content +"\n"*2+"Domains:"+ domain

        recommendations[jd_key] = {}
        recommendations[jd_key]["page_content"] = jd_data.page_content
        recommendations[jd_key]["summary"] = jd_summary
        recommendations[jd_key]["yoe"] = yoe
        recommendations[jd_key]["baseline"] = get_baseline_score(yoe)

        query_docs = text_splitter([jd_data])
        recommendations[jd_key]["docs"] = {}

        for resume_doc in resume_data:

            text_key = resume_doc.metadata["source"]
            recommendations[jd_key]["docs"][text_key] = {}

            recommendations[jd_key]["docs"][text_key]["page_content"] = resume_doc.page_content
            # recommendations[jd_key]["docs"][text_key]["page_content"] = "resume_doc.page_content"

            resume_text = " ".join(process_text(resume_doc.page_content))
            # recommendations[jd_key]["docs"][text_key]["processed_content"] = "processed_text"
            recommendations[jd_key]["docs"][text_key]["processed_content"] = resume_text


            #### Total Experience, Technical Experience, Domain Experience
            output = []
            llm_raw = []
            resume_content = resume_doc.page_content
            for i in range(0,len(resume_content),100000):
                cur = resume_content[max(i - 100,0):min(i+100000,len(resume_content))]
                # ans = "".join(get_experience_yaml(cur))

                if style == styles[0]:
                    temp = get_gemini_single_prompt(jd_summary,cur)
                    llm_raw.append(temp)
                    # ans = "".join(llm_raw[-1][0].text)
                    # output.append(ans)

                elif style == styles[1]:
                    temp = get_experience_gemini(jd_summary,cur)
                    llm_raw.append(temp)
                    ans = "".join(llm_raw[-1][0].text)
                    output.append(ans)



            if style == styles[0]:
                pass
                # recommendations[jd_key]["docs"][text_key]["llm_output"] = llm_raw
                # recommendations[jd_key]["docs"][text_key]["llm_text"] = "".join(output)
                # # experience = scan_all_experience(recommendations[jd_key]["docs"][text_key]["llm_text"])
                # # recommendations[jd_key]["docs"][text_key]["experience_list"] = experience[0]
                # # recommendations[jd_key]["docs"][text_key]["total_experience"] = experience[1]
                # # recommendations[jd_key]["docs"][text_key]["tech_experience"] = experience[2]
                # # recommendations[jd_key]["docs"][text_key]["domain_experience"] = experience[3]


                # #### Text-JD Cosine Score
                # text_docs = text_splitter([resume_doc])
                # score = query_text_cosine_score(query_docs,text_docs)
                # recommendations[jd_key]["docs"][text_key]["cosine_score"] = score


                # #### Composite Score
                # recommendations[jd_key]["docs"][text_key]["final_score"] = 0.0

                # # print(ans,len(ans))
                # # print(data.split("\n"))
                # # data = process_resume(data.page_content)
                # # import datefinder
                # # matches = datefinder.find_dates(data)
                # # for match in matches:
                # #     print(match)
                # # experience_all(data)
                # # print("\n"*4)


            elif style == styles[1]:
                recommendations[jd_key]["docs"][text_key]["llm_output"] = llm_raw
                recommendations[jd_key]["docs"][text_key]["llm_text"] = "".join(output)
                # experience = scan_all_experience(recommendations[jd_key]["docs"][text_key]["llm_text"])
                # recommendations[jd_key]["docs"][text_key]["experience_list"] = experience[0]
                # recommendations[jd_key]["docs"][text_key]["total_experience"] = experience[1]
                # recommendations[jd_key]["docs"][text_key]["tech_experience"] = experience[2]
                # recommendations[jd_key]["docs"][text_key]["domain_experience"] = experience[3]


                #### Text-JD Cosine Score
                text_docs = text_splitter([resume_doc])
                score = skills_match.query_text_cosine_score(query_docs,text_docs)
                recommendations[jd_key]["docs"][text_key]["cosine_score"] = score


                #### Composite Score
                recommendations[jd_key]["docs"][text_key]["final_score"] = 0.0

                # print(ans,len(ans))
                # print(data.split("\n"))
                # data = process_resume(data.page_content)
                # import datefinder
                # matches = datefinder.find_dates(data)
                # for match in matches:
                #     print(match)
                # experience_all(data)
                # print("\n"*4)


            elif style == styles[2]:
                text_docs = text_splitter([resume_doc])
                score = skills_match.query_text_cosine_score(query_docs,text_docs)
                recommendations[jd_key]["docs"][text_key]["cosine_score"] = score

                llm_raw,output = prompting.split_exp_score_prompt(jd_summary,resume_content)

                print("Prompting Done Successfully for {}!!!".format(resume_doc.metadata["source"]))

                recommendations[jd_key]["docs"][text_key]["llm_output"] = llm_raw
                recommendations[jd_key]["docs"][text_key]["llm_text_only"] = [(ele[0][0].text,ele[1][0].text) for ele in llm_raw]
                recommendations[jd_key]["docs"][text_key]["llm_text"] = "".join(output)
                recommendations[jd_key]["docs"][text_key]["exp_eval"] = parse_dates_matches(recommendations[jd_key]["docs"][text_key]["llm_text_only"],yoe)

                final_exp_score = recommendations[jd_key]["docs"][text_key]["exp_eval"]["final_score"]
                final_baseline_score = recommendations[jd_key]["baseline"]["final_score"]

                recommendations[jd_key]["docs"][text_key]["text_it"] = "og"
                if util.is_within_percentage(final_exp_score,final_baseline_score):
                    llm_raw,output = prompting.split_exp_score_prompt(jd_summary,resume_content)

                    recommendations[jd_key]["docs"][text_key]["alt_llm_output"] = llm_raw
                    recommendations[jd_key]["docs"][text_key]["alt_llm_text_only"] = [(ele[0][0].text,ele[1][0].text) for ele in llm_raw]
                    recommendations[jd_key]["docs"][text_key]["alt_llm_text"] = "".join(output)

                    print("Second Prompt Done Successfully for {}!!!".format(resume_doc.metadata["source"]))

                    recommendations[jd_key]["docs"][text_key]["alt_exp_eval"] = parse_dates_matches(recommendations[jd_key]["docs"][text_key]["alt_llm_text_only"],yoe)

                    if recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"] > final_exp_score:
                        final_exp_score = recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"]
                        recommendations[jd_key]["docs"][text_key]["text_it"] = "alt"
                    # else:
                    #     recommendations[jd_key]["docs"][text_key]["text_it"] = "both"
                    #     final_exp_score = (recommendations[jd_key]["docs"][text_key]["exp_eval"]["final_score"] + final_exp_score)/2


                recommendations[jd_key]["docs"][text_key]["final_score"] = final_exp_score
                recommendations[jd_key]["docs"][text_key]["recommendation"] = "E: Selected" if final_exp_score > final_baseline_score  else "E: Rejected"

            print("JD:",jd_key,"Doc:",text_key,"Done !!!")
            time.sleep(int(os.getenv("LONG_TIME")))
            print("\n"*2)
        # Uncomment for only single run
            break
        break


    # print(recommendations)
    util.save_pickle(recommendations,os.path.join(root_path, "dump","pickles",f"{args.filename}"))

'''

remove_path = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.dirname(remove_path)
file_path = os.path.join(remove_path,"pickles")
sys.path.remove(remove_path)
sys.path.insert(0,new_path)
from util import *

from util import load_pickle, save_pickle
from src.load_env import load_env_file,get_root_path

data = load_pickle(os.path.join(file_path,"info.pkl"))
for key in data.keys():
    # print(data[key])
    print(key,"\n")
    print("\n"*2)
    # input("")



load_env_file(".env")
info = {}
root_path = get_root_path()
projects_path = os.path.join(root_path,"docs","projects")
count = 0
for project in os.listdir(projects_path):
    if "DS_Store" in project:
        continue


    project_path = os.path.join(projects_path,project)
    jd_path = os.path.join(projects_path,project,"JD")
    resumes_path = os.path.join(projects_path,project,"resumes")


    jd_data = load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
    resume_data = load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

    # print([data.metadata["source"] for data in jd_data])
    # print([data.metadata["source"] for data in resume_data])
    print("\n"*2)


    # print(query_docs)

    for resume_doc in resume_data:
        # print(resume_doc)
        text_docs = text_splitter([resume_doc])
        print(resume_doc.metadata["source"], end = " ")
        score = query_text_cosine_score(query_docs,text_docs)
        print(score)

    # break

#         output = []
#         resume_content = resume_doc.page_content
#         for i in range(0,len(resume_content),3000):
#             cur = resume_content[max(i - 100,0):min(i+3000,len(resume_content))]
#             ans = "".join(get_experience_yaml(cur))
#             output.append(ans)
#             time.sleep(30)
#         info[resume_doc.metadata["source"]] = output
#         # break
#     time.sleep(60)
#     # break

save_pickle(info,os.path.join(file_path,"info_skills.pkl"))





if __name__=="__main__":
    pass

    # Experience Component

    load_env_file(".env")
    info = {}
    root_path = get_root_path()
    projects_path = os.path.join(root_path,"docs","projects")
    count = 0
    for project in os.listdir(projects_path):
        if "DS_Store" in project:
            continue

        project_path = os.path.join(projects_path,project)
        jd_path = os.path.join(projects_path,project,"JD")
        resumes_path = os.path.join(projects_path,project,"resumes")

        # Load Pickles
        jd_data = load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
        resume_data = load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

        # print([data.metadata["source"] for data in jd_data])
        # print([data.metadata["source"] for data in resume_data])
        # print("\n"*2)
        for resume_doc in resume_data:
            output = []
            resume_content = resume_doc.page_content
            for i in range(0,len(resume_content),3000):
                cur = resume_content[max(i - 100,0):min(i+3000,len(resume_content))]
                ans = "".join(get_experience_yaml(cur))
                output.append(ans)
                time.sleep(30)
            info[resume_doc.metadata["source"]] = output
            # break
        time.sleep(60)
        # break
    save_pickle(info,os.path.join(file_path,"info_exp.pkl"))


    load_env_file(".env")
    info = {}
    root_path = get_root_path()
    projects_path = os.path.join(root_path,"docs","projects")
    count = 0
    for project in os.listdir(projects_path):
        if "DS_Store" in project:
            continue


        project_path = os.path.join(projects_path,project)
        jd_path = os.path.join(projects_path,project,"JD")
        resumes_path = os.path.join(projects_path,project,"resumes")


        jd_data = load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
        resume_data = load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

        # print([data.metadata["source"] for data in jd_data])
        # print([data.metadata["source"] for data in resume_data])
        print("\n"*2)

        query_docs = text_splitter(jd_data)
        # print(query_docs)

        for resume_doc in resume_data:
            # print(resume_doc)
            text_docs = text_splitter([resume_doc])
            print(resume_doc.metadata["source"], end = " ")
            score = query_text_cosine_score(query_docs,text_docs)
            print(score)

        # break

    #         output = []
    #         resume_content = resume_doc.page_content
    #         for i in range(0,len(resume_content),3000):
    #             cur = resume_content[max(i - 100,0):min(i+3000,len(resume_content))]
    #             ans = "".join(get_experience_yaml(cur))
    #             output.append(ans)
    #             time.sleep(30)
    #         info[resume_doc.metadata["source"]] = output
    #         # break
    #     time.sleep(60)
    #     # break

    save_pickle(info,os.path.join(file_path,"info_skills.pkl"))



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


'''
