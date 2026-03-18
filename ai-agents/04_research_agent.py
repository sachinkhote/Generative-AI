import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load HR Policy
print("📄 Loading HR Policy...")
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
    persist_directory="./chroma_db_research"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Ready!")

# Tool 1 — HR Policy Search
@tool
def hr_policy_search(query: str) -> str:
    """Search the HR Policy document for information about
    leaves, attendance, salary, recruitment, probation,
    promotion, dress code, grievance, retirement and other
    HR policies."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in HR policy."
    return "\n\n".join(doc.page_content for doc in docs)

# Tool 2 — Web Search
search_tool = TavilySearch(
    max_results=3,
    topic="general"
)

# Tool 3 — Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate any mathematical expression.
    Input should be a valid math expression like '15 * 8500 / 100'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Tool 4 — Save to File
@tool
def save_report(filename: str, content: str) -> str:
    """Save research findings or any content to a text file.
    Use this when user asks to save, export or create a report.
    Input: filename (e.g. 'report.txt') and content to save."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        full_content = f"Report Generated: {timestamp}\n{'='*50}\n\n{content}"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        return f"✅ Report saved successfully as '{filename}'"
    except Exception as e:
        return f"Error saving file: {e}"

# All tools
tools = [hr_policy_search, search_tool, calculator, save_report]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent research assistant with access to these tools:
1. tavily_search — search the web for current information
2. hr_policy_search — search HR policy document
3. calculator — perform mathematical calculations
4. save_report — save content to a file

Guidelines:
- Use tavily_search for current news, facts, prices
- Use hr_policy_search for HR policy questions
- Use calculator for any math
- Use save_report when user asks to save or create a report
- Combine multiple tools when needed
- Always provide comprehensive answers"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5  # prevent infinite loops
)

print("="*50)
print("   Research Agent 🔬")
print("="*50)
print("I can search, calculate and save reports!")
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

    # Update history
    chat_history.append({"role": "human", "content": user_input})
    chat_history.append({"role": "assistant", "content": clean_output})

    print(f"\n🤖 Agent: {clean_output}\n")
    print("-"*50)