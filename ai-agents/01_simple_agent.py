import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent       # runs agent loop (thought -> action -> Observation) , creates agent that knows how to use tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool                                               # converts a normal python function into an agent tool

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Define tools — these are functions the agent can call
@tool                                               # this tells Langchain - this function is a tool the agent can use ,- LLM reads this to decide when to use it.
def calculator(expression: str) -> str:
    """Calculate any mathematical expression. 
    Input should be a valid math expression like '15 * 8500 / 100'"""       #LLM reads this description to decide
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"The text has {count} words."

@tool
def reverse_text(text: str) -> str:
    """Reverse any given text."""
    return text[::-1]

# List of tools
tools = [calculator, word_counter, reverse_text]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.
Use tools whenever needed to give accurate answers.
Always use the calculator tool for math — never calculate mentally."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")           # agent_scratchpad = the agent's working memory — where it writes its thoughts and tool results during reasoning
])                                                                  # rough paper for agents

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)               # wires llm +tools + prompt
agent_executor = AgentExecutor(                                     #  the engine that runs the Thought → Action → Observation loop
    agent=agent,
    tools=tools,
    verbose=True  # ← shows agent's thinking process!
)

print("="*50)
print("   Simple AI Agent 🤖")
print("="*50)
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

    print(f"\n🤖 Agent: {response['output']}\n")
    print("-"*50)