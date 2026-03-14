import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate               # langchain classs for creating reusable prompt templates

# Load API key
load_dotenv()

# Connect to Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3                                         # low tempearture coz, we want consistent, professional summaries — not creative ones
)

# Create a reusable prompt template
template = PromptTemplate(
    input_variables=["job_title", "experience", "skills"],
    template="""
You are a career advisor.
Summarize this job role in exactly 2 sentences.
Keep it simple and professional.

Job Title: {job_title}
Experience Required: {experience}
Key Skills: {skills}
"""
)

# Use template with different data
jobs = [
    {
        "job_title": "GenAI Developer",
        "experience": "Fresher",
        "skills": "Python, LangChain, RAG, LLM APIs"
    },
    {
        "job_title": "Data Analyst",
        "experience": "1-2 years",
        "skills": "SQL, Python, Power BI, Excel"
    },
    {
        "job_title": "ML Engineer",
        "experience": "2-3 years",
        "skills": "Python, TensorFlow, PyTorch, MLOps"
    }
]                                                                      # this is just python lis of dictionary, these values will be injected into template 

print("="*50)
print("   Job Summarizer using PromptTemplate")
print("="*50)

for job in jobs:                                                    # loops though jobs one by one
    # Fill template with job data
    prompt = template.format(**job)                                 # fills the template with job details
    
    # Send to LLM
    response = llm.invoke(prompt)
    
    print(f"\n🎯 {job['job_title']}:")
    print(f"   {response.content}")
    print("-"*50)