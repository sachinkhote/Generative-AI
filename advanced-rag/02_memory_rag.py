import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2                     # for controlled and professional response
)   

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"               # this model converts text into numerical vectors
)

# Load and chunk PDF
print("📄 Loading PDF...")
loader = PyPDFLoader("NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,                        # creates chunk of 1000 characters with 100 overlap
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_memory"      # stores vectors at this location
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})    # it will fetch top3 most relevent chunks

# RAG prompt WITH conversation history
prompt = ChatPromptTemplate.from_messages([                 # prompt for the llm 
    ("system", """You are an HR Policy Assistant.
Answer questions based ONLY on the provided context.
If answer is not in context say: 
'I couldn't find this in the HR policy. Please contact HR.'
Keep answers concise and professional."""),
    MessagesPlaceholder(variable_name="chat_history"),  # ← memory
    ("human", """Context: {context}

Question: {question}""")
])

parser = StrOutputParser()              # to extract string response from llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)        # helper function that joins the page content of retrived documents

# Conversation history
chat_history = []               # empty list to store chat conversation( will hold HumanMessage and AIMessage objects)

print("✅ HR Policy Assistant with Memory Ready!")
print("="*50)
print("   HR Policy Assistant 🤖 + Memory 🧠")
print("="*50)
print("Type 'exit' to quit.\n")

# Query contextualizer
contextualizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the conversation history and latest question,
rewrite the question to be self-contained and searchable.
If no history, return question as is.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

contextualizer = contextualizer_prompt | llm | parser           # chains this to form a contextualizer

while True:                                             # check for user i/p
    question = input("You: ")

    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    if question.strip() == "":
        continue

    # Step 1 — Rewrite question using history
    if chat_history:                                # if chat history exists, it invokes contextualizer to rewrite the question
        search_query = contextualizer.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print(f"🔍 Searching for: {search_query}")
    else:
        search_query = question

    # Step 2 — Retrieve using rewritten question
    docs = retriever.invoke(search_query)           # to fetch relevent document based on search query
    context = format_docs(docs)                     # then formats them into contexxt

    # Step 3 — Generate answer with memory
    response = prompt | llm | parser                # chains the RAG prompt with LLM and parser
    answer = response.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    # Step 4 — Update memory
    chat_history.append(HumanMessage(content=question))     # appends user question and AI answer to chat historyx
    chat_history.append(AIMessage(content=answer))

    print(f"\n🤖 Assistant: {answer}\n")
    print("-"*50)