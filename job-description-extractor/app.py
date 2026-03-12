import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("Key found:", os.getenv("GEMINI_API_KEY"))

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Load the model
model = genai.GenerativeModel("gemini-2.5-flash")

print(" Gemini connected successfully!")

def extract_job_details(job_description):
    
    prompt = f"""
You are a job description analyzer.
Extract information and return ONLY valid JSON.
No explanation. No extra text. JSON only.

Return in this exact format:
{{
  "job_title": "",
  "experience_required": "",
  "skills": [],
  "salary": "Not mentioned",
  "job_type": "",
  "summary": ""
}}

Here are 2 examples:

Job: "We are hiring a Python Developer with 2 years experience. 
Skills needed: Python, Django, REST APIs. Salary: 8-10 LPA. Full time role."

Output: {{"job_title": "Python Developer", "experience_required": "2 years", 
"skills": ["Python", "Django", "REST APIs"], "salary": "8-10 LPA", 
"job_type": "Full time", "summary": "Python developer role focused on web development"}}

Job: "Looking for Data Analyst fresher. Must know SQL, Excel, Power BI. 
Work from home position. Salary not disclosed."

Output: {{"job_title": "Data Analyst", "experience_required": "Fresher", 
"skills": ["SQL", "Excel", "Power BI"], "salary": "Not mentioned", 
"job_type": "Work from home", "summary": "Fresher data analyst role with BI tools focus"}}

Now extract from this job description:
{job_description}
"""

    # Send to Gemini
    response = model.generate_content(prompt)
    
    # Get the text response
    raw_output = response.text
    # Clean markdown code blocks if present
    raw_output = raw_output.strip().replace("```json", "").replace("```", "").strip()
    
    # Parse JSON
    result = json.loads(raw_output)
    
    return result

print("\n" + "="*50)
print("   Job Description Extractor 🔍")
print("   Powered by Gemini + Prompt Engineering")
print("="*50)
print("Paste a job description and press Enter twice.\n")

while True:
    print("Paste job description (or type 'exit'):")
    
    # Collect multi-line input
    lines = []
    while True:
        line = input()
        if line.lower() == "exit":
            print("\nGoodbye! 👋")
            exit()
        if line == "":
            break
        lines.append(line)
    
    job_description = " ".join(lines)
    
    if not job_description.strip():
        continue
    
    print("\n⏳ Extracting details...")
    
    result = extract_job_details(job_description)
    
    print("\n✅ Extracted Details:")
    print("="*40)
    print(f"🎯 Job Title     : {result['job_title']}")
    print(f"📅 Experience    : {result['experience_required']}")
    print(f"💰 Salary        : {result['salary']}")
    print(f"💼 Job Type      : {result['job_type']}")
    print(f"🛠️  Skills        : {', '.join(result['skills'])}")
    print(f"📝 Summary       : {result['summary']}")
    print("="*40 + "\n")