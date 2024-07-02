# Resume Scanner

## Description

The Resume Scanner project automates the matching of job requirements from a given job description file with the skills and experience listed in multiple resume files. It utilizes LLM (Gemini API) for natural language processing tasks. The goal is to provide an ordered list of resumes based on their relevance to the job description.

## Project Requirements

- **Accuracy and Repeatability:** Ensure consistent and accurate scoring of resumes against job descriptions.
- **Performance Metrics:**
  - **90% Recall:** Identify candidates suitable for technical interviews.
  - **40% Precision:** Ensure 40% - 50% of selected candidates advance to final or prefinal rounds.

## Technology Stack

- **Python:** Main programming language for development.
- **Langchain:** Framework used for implementing natural language processing tasks.
- **geminiai:** Utilized for integrating with LLM (Gemini API).

## Expected Timeline

Development of the prototype: 45 days.

## Workflow

1. **Job Description Parsing:**
   - Reads the job description file and extracts technical requirements and minimum required years of experience.
   - Prompts LLM (Gemini API) to store this information in variables.

2. **Resume Processing:**
   - Iterates through each resume file.
   - Passes the extracted job requirements to LLM to extract experiences and their start and end dates as YAML.

3. **Matching Process:**
   - Uses LLM (Gemini API) to retrieve experiences from resumes as YAML.
   - Matches these experiences against the technical and domain scores derived from the job description.

4. **Output Formatting:**
   - Formats the matched data into a Python dictionary.
   - Stores the formatted data in a pickle file.

5. **Result Display:**
   - Loads the pickle file and prints the matched data for review.
   - [Sample Output](sample_op.txt)

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd resume-scanner
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage Example

```shell
../python ../ResumeScanner/main.py --filename="trial.pkl" > text_main.txt
../python ../ResumeScanner/print.py --filename="trial.pkl" > text_print.txt
```

## Future Features

- **Customizable Output and Input:** Modify output formats or add visualization options.
- **Error Handling:** Implement robust error handling for file formats and API responses.
- **Performance Optimization:** Enhance processing speed for large datasets.
- **User Interface:** Develop a simple GUI for easier interaction.

## License

This project is licensed under the Apache License 2.0