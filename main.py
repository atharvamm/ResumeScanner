import os
import time
from src.load_env import load_env_file, get_root_path
import src.parse_doc as parse_doc
import src.util as util
from src.process_text import process_text
import src.experience as experience
import src.skills_match as skills_match
import src.prompting as prompting
import argparse
import datetime

# Main function to execute the resume scanning and matching process
if __name__ == "__main__":

    # Setting up argument parser to accept filename as a command-line argument
    parser = argparse.ArgumentParser(description="Print for resume scanner.")
    parser.add_argument("--filename", required=True,
                        help="The name of the file to process")
    args = parser.parse_args()
    
    # https://stackoverflow.com/questions/3316882/how-do-i-get-a-string-format-of-the-current-date-time-in-python
    # Print the current date and time
    print("Run:", datetime.datetime.now().strftime("%I:%M:%S%p on %B %d, %Y"))
    print(f"Running: {args.filename}")


    # Load environment variables from .env file
    load_env_file(".env")

    # Get the root path of the project
    root_path = get_root_path()
    projects_path = os.path.join(root_path, "docs", "projects")
    count = 0

    # Initialize variables
    recommendations = {}

    styles = ["single_prompt", "single_cot", "breakexpandmatch"]
    # Single Prompt: Write a single prompt to find experiences and also calculate coressponding scores.
    # Single COT: Get experiences and match from LLM and then parse the output text to get relevant information.
    # BreakExpandMatch: Break prompt to get experience from resume and match scores relative to job description.
    style = styles[2] # Select the prompting style to use

    # Iterate through each project directory in the projects_path
    for project in os.listdir(projects_path):
        if "DS_Store" in project: # Skip MacOS System Files
            continue
        print("Current Project: ", project)

        if "nodejs2" not in project:
            continue

        # Path to job description, resumes.
        project_path = os.path.join(projects_path, project)
        jd_path = os.path.join(projects_path, project, "JD")
        resumes_path = os.path.join(projects_path, project, "resumes")

        # Parse job descriptions and resumes
        jd_data = parse_doc.parse_dir(jd_path)
        resume_data = parse_doc.parse_dir(resumes_path)

        # Save pickle files with resume and job description data
        util.save_pickle(jd_data, os.path.join(
            root_path, "dump", "pickles", f"jd_data_{project}.pkl"))
        util.save_pickle(resume_data, os.path.join(
            root_path, "dump", "pickles", f"resume_data_{project}.pkl"))

        # Load pickle files with resume and job description data
        jd_data = util.load_pickle(os.path.join(
            root_path, "dump", "pickles", f"jd_data_{project}.pkl"))
        resume_data = util.load_pickle(os.path.join(
            root_path, "dump", "pickles", f"resume_data_{project}.pkl"))

        # Store job description metadata
        jd_data = jd_data[0]
        jd_key = jd_data.metadata["source"]

        # Job Description Summary and YOE
        domain, yoe = prompting.get_jd_domain_reqs(
            jd_data.page_content, func_type="prompt")
        jd_summary = jd_data.page_content + "\n"*2+"Domains:" + domain

        # Store final job description data
        recommendations[jd_key] = {}
        recommendations[jd_key]["page_content"] = jd_data.page_content
        recommendations[jd_key]["summary"] = jd_summary
        recommendations[jd_key]["yoe"] = yoe
        recommendations[jd_key]["baseline"] = experience.get_baseline_score(
            yoe)
        final_baseline_score = recommendations[jd_key]["baseline"]["final_score"]

        query_docs = parse_doc.text_splitter([jd_data])
        recommendations[jd_key]["docs"] = {}

        # Iterate through resumes
        for resume_doc in resume_data:

            text_key = resume_doc.metadata["source"]

            recommendations[jd_key]["docs"][text_key] = {}

            recommendations[jd_key]["docs"][text_key]["page_content"] = resume_doc.page_content

            resume_text = " ".join(process_text(resume_doc.page_content))
            recommendations[jd_key]["docs"][text_key]["processed_content"] = resume_text
            resume_content = resume_doc.page_content

            # Get cosine score for job description and resume
            text_docs = parse_doc.text_splitter([resume_doc])
            score = skills_match.query_text_cosine_score(query_docs, text_docs)
            recommendations[jd_key]["docs"][text_key]["cosine_score"] = score

            if style == styles[0]:
                llm_raw, output = prompting.singleprompt(
                    jd_summary, resume_content)

            elif style == styles[1]:
                llm_raw, output = prompting.single_cot(
                    jd_summary, resume_content)

                print("Prompting Done Successfully for {}!!!".format(
                    resume_doc.metadata["source"]))

                recommendations[jd_key]["docs"][text_key]["llm_output"] = llm_raw
                recommendations[jd_key]["docs"][text_key]["llm_text"] = "".join(
                    output)

                info = experience.parse_llmop_style1(
                    recommendations[jd_key]["docs"][text_key]["llm_text"])
                info = experience.order_exp(info)
                recommendations[jd_key]["docs"][text_key]["exp_eval"] = experience.score_exp(
                    info, yoe)

                final_exp_score = recommendations[jd_key]["docs"][text_key]["exp_eval"]["final_score"]

                recommendations[jd_key]["docs"][text_key]["text_it"] = "og"
                if util.is_within_percentage(final_exp_score, final_baseline_score):
                    llm_raw, output = prompting.single_cot(
                        jd_summary, resume_content)

                    recommendations[jd_key]["docs"][text_key]["alt_llm_output"] = llm_raw
                    recommendations[jd_key]["docs"][text_key]["alt_llm_text"] = "".join(
                        output)

                    print("Second Prompt Done Successfully for {}!!!".format(
                        resume_doc.metadata["source"]))

                    info = experience.parse_llmop_style1(
                        recommendations[jd_key]["docs"][text_key]["alt_llm_text"])
                    info = experience.order_exp(info)
                    recommendations[jd_key]["docs"][text_key]["alt_exp_eval"] = experience.score_exp(
                        info, yoe)

                    if recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"] > final_exp_score:
                        final_exp_score = recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"]
                        recommendations[jd_key]["docs"][text_key]["text_it"] = "alt"

            elif style == styles[2]:
                llm_raw, output = prompting.split_exp_score_prompt(
                    jd_summary, resume_content)

                print("Prompting Done Successfully for {}!!!".format(
                    resume_doc.metadata["source"]))

                recommendations[jd_key]["docs"][text_key]["llm_output"] = llm_raw
                recommendations[jd_key]["docs"][text_key]["llm_text_only"] = [
                    (ele[0][0].text, ele[1][0].text) for ele in llm_raw]
                recommendations[jd_key]["docs"][text_key]["llm_text"] = "".join(
                    output)

                info = experience.parse_llmop_style2(
                    recommendations[jd_key]["docs"][text_key]["llm_text_only"])
                info = experience.order_exp(info)
                recommendations[jd_key]["docs"][text_key]["exp_eval"] = experience.score_exp(
                    info, yoe)

                final_exp_score = recommendations[jd_key]["docs"][text_key]["exp_eval"]["final_score"]

                recommendations[jd_key]["docs"][text_key]["text_it"] = "og"
                # If score is close to baseline pass the resume to 
                if util.is_within_percentage(final_exp_score, final_baseline_score):
                    llm_raw, output = prompting.split_exp_score_prompt(
                        jd_summary, resume_content)

                    recommendations[jd_key]["docs"][text_key]["alt_llm_output"] = llm_raw
                    recommendations[jd_key]["docs"][text_key]["alt_llm_text_only"] = [
                        (ele[0][0].text, ele[1][0].text) for ele in llm_raw]
                    recommendations[jd_key]["docs"][text_key]["alt_llm_text"] = "".join(
                        output)

                    print("Second Prompt Done Successfully for {}!!!".format(
                        resume_doc.metadata["source"]))

                    info = experience.parse_llmop_style2(
                        recommendations[jd_key]["docs"][text_key]["alt_llm_text_only"])
                    info = experience.order_exp(info)
                    recommendations[jd_key]["docs"][text_key]["alt_exp_eval"] = experience.score_exp(
                        info, yoe)

                    if recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"] > final_exp_score:
                        final_exp_score = recommendations[jd_key]["docs"][text_key]["alt_exp_eval"]["final_score"]
                        recommendations[jd_key]["docs"][text_key]["text_it"] = "alt"

            # Final evaluation and recommendation for the resume
            recommendations[jd_key]["docs"][text_key]["final_score"] = final_exp_score
            recommendations[jd_key]["docs"][text_key]["recommendation"] = "E: Selected" if final_exp_score > final_baseline_score else "E: Rejected"

            print("JD:", jd_key, "Doc:", text_key, "Done !!!")
            time.sleep(int(os.getenv("LONG_TIME")))
            print("\n"*2)

    # Save the final recommendations to a pickle file
    util.save_pickle(recommendations, os.path.join(
        root_path, "dump", "pickles", f"{args.filename}"))
