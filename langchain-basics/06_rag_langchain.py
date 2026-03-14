import os                                                               # allows interaction with os,Retrieves environment variables 
from dotenv import load_dotenv                                          # reads hidden files
from langchain_google_genai import ChatGoogleGenerativeAI               # connects to gemini and processes prompts and generates response 
from langchain_community.document_loaders import PyPDFLoader            # extracts text from pdf files
from langchain_text_splitters import RecursiveCharacterTextSplitter     # divides pdf text into smaller chunks (paragraphs)
from langchain_huggingface import HuggingFaceEmbeddings                 # converts text to numerical vectors which represents meaning of words
from langchain_chroma import Chroma                                     # vector store saves the numbered chunks
from langchain_core.prompts import ChatPromptTemplate                   # creates template to communicate with AI
from langchain_core.output_parsers import StrOutputParser               # processes final answer from AI and presents it as readable string 

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# Embedding Model (free, runs locally, no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("✅ LLM and Embeddings ready!")

# Block 2 - Load and chunk PDF
print("\n📄 Loading PDF...")
loader = PyPDFLoader("NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Block 3 - Store chunks in ChromaDB
print("\n🗄️ Creating Vector Database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # saves locally on your machine
)
print("✅ Vector Database created and saved!")

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # retrieve top 3 relevant chunks
)
print("✅ Retriever ready!")

# Block 4 - Create RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR Policy Assistant.
Answer questions based ONLY on the provided context.
If answer is not in context, say: 
'I couldn't find this in the HR policy. Please contact HR.'
Keep answers concise and professional."""),
    ("human", """Context: {context}

Question: {question}""")
])

# Output parser
parser = StrOutputParser()

# Block 5 - Build the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | parser
)

# Block 6 - Interactive loop
print("\n" + "="*50)
print("   HR Policy Assistant 🤖")
print("   Powered by LangChain + ChromaDB")
print("="*50)
print("Ask anything about HR policy!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    if question.strip() == "":
        continue

    answer = rag_chain.invoke(question)
    print(f"\n🤖 Assistant: {answer}\n")
    print("-"*50)