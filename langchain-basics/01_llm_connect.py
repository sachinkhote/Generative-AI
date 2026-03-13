import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

# Connect to Gemini via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

Question = input("Ask Your Question or Query...: ")
# Send a simple message
response = llm.invoke(Question)

print("✅ LangChain connected!")
print("\n🤖 Response:", response.content)
