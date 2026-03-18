import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load HR Policy PDF
print(" Loading HR Policy...")
loader = PyPDFLoader(r"D:\Projects\GenAI Study Path\langchain-basics\NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_agent"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("HR Policy loaded!")

from langchain_tavily import TavilySearch

# RAG Tool — searches HR Policy document
@tool
def hr_policy_search(query: str) -> str:
    """Search the HR Policy document for information about
    leaves, attendance, salary, recruitment, probation,
    promotion, dress code, grievance, retirement and other
    HR policies. Use this for any HR policy related questions."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in HR policy."
    return "\n\n".join(doc.page_content for doc in docs)

# Web search tool
search_tool = TavilySearch(
    max_results=3,
    topic="general"
)

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate any mathematical expression.
    Input should be a valid math expression like '15 * 8500 / 100'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# All tools together
tools = [hr_policy_search, search_tool, calculator]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to these tools:
1. hr_policy_search — for ANY questions about HR policies, leaves, attendance, salary etc.
2. tavily_search — for current news, prices, facts from the internet
3. calculator — for mathematical calculations

Rules:
- ALWAYS use hr_policy_search for HR policy questions
- ALWAYS use tavily_search for current information
- ALWAYS use calculator for math
- Combine multiple tools when needed
- Never answer from memory alone"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

print("="*50)
print("   HR Policy + Web Agent 🤖")
print("="*50)
print("I can answer HR policy AND web questions!")
print("Type 'exit' to quit.\n")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    if user_input.strip() == "":
        continue

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Clean output
    output = response['output']
    if isinstance(output, list):
        clean_output = ""
        for item in output:
            if isinstance(item, dict) and 'text' in item:
                clean_output += item['text']
            elif isinstance(item, str):
                clean_output += item
    else:
        clean_output = str(output)

    print(f"\n🤖 Agent: {clean_output}\n")
    print("-"*50)