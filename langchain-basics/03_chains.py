import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model ="gemini-2.5-flash", temperature= 0.3)

prompt= PromptTemplate(input_variable =['topic'], 
                       template =''' You are a tech educator. Explain {topic} in exactly 3 simple sentences.
                       Use simple words a beginner can undersatnd.''')

parser =StrOutputParser()                   # just extracts clean text from response

chain = prompt | llm | parser

print('-'*50)
print("Langchain Chains Demo")
print('-'*50)
'''
topics =['RAG', 'Vector Database','LLM']

for topic in topics:
    response = chain.invoke({'topic':topic})
    print(f"\n {topic}:")
    print(f"\n {response}:")
'''
print("Ask me to explain any tech topic!")
print("Type 'exit' to quit.\n")

while True:
    topic = input("You: ")                              # for taking custom input from user
    
    if topic.lower() == "exit":
        print("\nGoodbye! 👋")
        break
    
    if topic.strip() == "":
        continue
    # Split by comma if user types multiple topics
    topics = [t.strip() for t in topic.split(",")]      # when we pass multiple topics at once
    
    for topic in topics:
        response = chain.invoke({"topic": topic})
        print(f"\n📚 {topic}:")
        print(f"   {response}")
        print("-"*50)