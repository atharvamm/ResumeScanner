from src.load_env import load_env_file, get_root_path
from src.parse_doc import parse_dir
from src.vector_store import get_vectorstore,get_connection,get_embedding
from src.process_text import process_text
from src.experience import *
from src.misc import misc_prompt
from src.skills_match import *
import os
import time


load_env_file(".env")

projects_path = os.path.join(get_root_path(),"docs","projects")
count = 0
for project in os.listdir(projects_path):
    if "DS_Store" in project:
        continue
    project_path = os.path.join(projects_path,project)
    jd_path = os.path.join(projects_path,project,"JD")
    resumes_path = os.path.join(projects_path,project,"resumes")
    # print(project_path,jd_path,resumes_path)
    jd_data = parse_dir(jd_path)
    resume_data = parse_dir(resumes_path)


    # print(jd_data)
    # print(resume_data)
    # recommendations = {}
    for data in jd_data:
        # print(data)
        # print(data.page_content[:40],data.metadata["source"])
        print(data.metadata["source"]) 
        jd_skills = pick_set(data.page_content.lower())
        jd_skills_keypoints = "".join(jd_prompt(jd_skills))
        time.sleep(20)

        # print(data.page_content.lower(),process_jd(data.page_content))
        # print(data.page_content)
        # ans = spacy(data.page_content)
        
        # ans = process_jd(data.page_content)
        # ans = " ".join(process_jd(data.page_content))
        # for ele in ans:
        #     pass
            # print(ele,str(ele),check_non_numeric(str(ele)))
            # if check_non_numeric(str(ele)):
            #     print(ele)
        # print(spacy(ans),len(spacy(ans)))
        # print(yake(ans),len(yake(ans)))
        # print(rake_nltk(ans),len(rake_nltk(ans)))

        # print(get_spacy_keybert(data.page_content))

    for data in resume_data:
        # count += 1
        # if count != 5:
        #     continue
        # print(count,data.metadata["source"],end = " ")
        # data = process_resume(data.page_content)
        # print(len(data))
        # print(data)
        print(data.metadata["source"],"\n")
        data = " ".join(process_text(data.page_content))
        # print(data)


        ## Run for Algorithm Skills
        # resume_skills = pick_set(data)
        # ans = rag_novectorstore_skills(query=jd_skills,text = resume_skills)
        # print(ans)
        ## Run for Algorithm Skills


        # ## Run for LLM Skills
        # resume_skills = pick_set(data)
        # print(data,len(data))
        prev = " "
        for i in range(0,len(data),2000):
            cur = data[max(i - 100,0):min(i+2000,len(data))]
            prev = "".join(skills_prompt(jd_skills_keypoints, prev,cur))
            time.sleep(45)
        print(jd_skills_keypoints)
        print("Final:",prev)
        # ## Run for LLM Skills


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
        time.sleep(120)
        # break
    #     # print(data.page_content[:40],data.metadata["source"])
    #     print(data.metadata["source"])
        # pass
    # break

# process_jd(job_description)



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


