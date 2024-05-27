from openai import OpenAI
import os


def misc_prompt(prev,cur):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "{}".format(os.getenv("NVIDIA_KEY"))
    )
    
    # text = "{}\n\n Extract the beginning, end, and organization of experiences from the resume and provide them as a list of tuples (Begin,End, Org). Output only the list and nothing else.".format(text)

    # text = '''
    # Previous Response:
    # {}

    # Current Resume Section:
    # {}

    # Judge the candidate on:
    # 1. English Language Proficiency : <Answer>
    # 2. Project Management Experience : <Answer>
    # 3. Achievements and Contributions : <Answer>
    # 4. Soft Skills : <Answer>
    # 5. Certifications and Training : <Answer>

    # Please combine your response to the previous resume section with the current resume section and provide detailed responses to each of the five questions listed above Ensure the format is same basically the answer should start from the same line as the colon and should be a pagraph.
    # '''.format(prev, cur)

    text = '''
    Judge the candidate on the following criteria and format the answer exactly as specified:
    1. English Language Proficiency: 
    2. Project Management Experience: 
    3. Achievements and Contributions: 
    4. Soft Skills: 
    5. Certifications and Training: 

    Previous Response:
    {}

    Current Resume Section:
    {}

    Please combine your response to the previous resume section with the current resume section and provide detailed responses to each of the five questions listed above. Only include answers to the questions; do not add any other extra text.
    '''.format(prev, cur)


    completion = client.chat.completions.create(
    model="meta/llama2-70b",
    messages=[{"role":"user","content":"{}".format(text)}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
    )
    output = []
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            output.append(chunk.choices[0].delta.content)

    return output