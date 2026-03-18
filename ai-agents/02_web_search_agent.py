import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Web search tool
search_tool = TavilySearch(
    max_results=3,
    topic="general"
)

# Calculator tool from before
@tool
def calculator(expression: str) -> str:
    """Calculate any mathematical expression.
    Input should be a valid math expression like '15 * 8500 / 100'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Tools list
tools = [search_tool, calculator]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant.
You have access to web search and calculator tools.
- Use web search for current information, news, facts
- Use calculator for any mathematical calculations
- Always search before answering questions about current events
- Combine multiple tool results when needed"""),
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
print("   Web Search Agent ")
print("="*50)
print("I can search the web and do calculations!")
print("Type 'exit' to quit.\n")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nGoodbye! ")
        break

    if user_input.strip() == "":
        continue

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Clean the output
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