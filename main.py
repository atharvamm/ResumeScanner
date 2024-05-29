from src.load_env import load_env_file, get_root_path
from src.parse_doc import parse_dir
from src.vector_store import get_vectorstore,get_connection,get_embedding
from src.process_text import process_text
from src.experience import *
from src.misc import misc_prompt
from src.skills_match import *
import os
import time
from util import save_pickle,load_pickle


load_env_file(".env")

root_path = get_root_path()
projects_path = os.path.join(root_path,"docs","projects")
count = 0
for project in os.listdir(projects_path):
    if "DS_Store" in project:
        continue

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
    jd_data = load_pickle(os.path.join(root_path, "dump","pickles",f"jd_data_{project}.pkl"))
    resume_data = load_pickle(os.path.join(root_path, "dump","pickles",f"resume_data_{project}.pkl"))

    # print([data.metadata["source"] for data in jd_data])
    # print([data.metadata["source"] for data in resume_data])
    # print("\n"*2)

    recommendations = {}
    for jd_doc in jd_data:

        jd_key = jd_doc.metadata["source"]
        recommendations[jd_key] = {}
        jd_skills = pick_set(jd_doc.page_content.lower())
        jd_skills_keypoints = jd_prompt(jd_skills)
        recommendations[jd_key]["jd_skills_keypoints"] = jd_skills_keypoints
        print(jd_key)
        print(jd_skills_keypoints)
        # break
        time.sleep(20)

        recommendations[jd_key]["docs"] = {}
        for resume_doc in resume_data:

            text_key = resume_doc.metadata["source"]
            resume_doc = " ".join(process_text(resume_doc.page_content))


            ## Run for Algorithm Skills
            # resume_skills = pick_set(data)
            # ans = rag_novectorstore_skills(query=jd_skills,text = resume_skills)
            # print(ans)
            ## Run for Algorithm Skills


            # # ## Run for LLM Skills
            # # resume_skills = pick_set(data)
            # # print(data,len(data))
            # recommendations[jd_key]["docs"][text_key] = []
            # for i in range(0,len(resume_doc),3000):
            #     cur = resume_doc[max(i - 100,0):min(i+3000,len(resume_doc))]
            #     ans = "".join(skills_prompt(jd_skills_keypoints,cur))
            #     recommendations[jd_key]["docs"][text_key].append(ans)
            #     time.sleep(30)
            # print(jd_key,text_key)
            # print(jd_skills_keypoints)
            # print("Final:",recommendations[jd_key]["docs"][text_key])
            # # ## Run for LLM Skills


            # ## Run for Algorithm Experience
            # ans = clean_exp(data)
            # print(experience_algo(ans))
            # ## Run for Algorithm Experience


            # ## Run for LLM Experience
            # ans = clean_exp_llm(data)
            # print(ans)
            # print(experience_llm(ans))
            # ## Run for LLM Experience


            # ## Run for Misc
            # print(data,len(data))
            # prev = " "
            # for i in range(0,len(data),2500):
            #     cur = data[max(i - 100,0):min(i+2500,len(data))]
            #     prev = "".join(misc_prompt(prev,cur))
            #     # print("Prev:",prev)
            #     # print("Cur:",cur)
            #     time.sleep(45)
            # print("Final:",prev)
            # ## Run for Misc


            # print(ans,len(ans))
            # print(data.split("\n"))
            # data = process_resume(data.page_content)
            # import datefinder
            # matches = datefinder.find_dates(data)
            # for match in matches:
            #     print(match)
            # experience_all(data)
            print("\n"*4)
            time.sleep(60)
    #         break
    #     break
    # break
save_pickle(recommendations,os.path.join(root_path, "dump","pickles","recommendations.pkl"))


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


