import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load and chunk PDF
print(" Loading PDF...")
loader = PyPDFLoader("NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f" {len(chunks)} chunks created")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Normal retriever
normal_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

# Multi-query retriever
# It uses LLM to generate multiple versions of the query!
from langchain_core.prompts import PromptTemplate

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 3 different versions of the given question 
to search an Indian HR policy document. 
Use terms like: leave, CL, casual leave, LWP, vacation, policy.
Return only the 3 questions, one per line, no numbering.

Original question: {question}"""
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=normal_retriever,
    llm=llm,
    prompt=multi_query_prompt
)

# RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR Policy Assistant.
Answer ONLY based on the provided context.
If not found say: 'I couldn't find this in the HR policy.'"""),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

parser = StrOutputParser()

def format_docs(docs):
    # Remove duplicate chunks
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return "\n\n".join(doc.page_content for doc in unique_docs)

# Two chains for comparison
normal_chain = (
    {"context": normal_retriever | format_docs, "question": lambda x: x}
    | prompt | llm | parser
)

multi_chain = (
    {"context": multi_retriever | format_docs, "question": lambda x: x}
    | prompt | llm | parser
)

# Compare both on same question
question = "How many leaves do employees get?"

print("\n" + "="*50)
print("COMPARISON: Normal vs Multi-Query Retrieval")
print("="*50)

print(f"\n Question: {question}")

print("\n Normal Retriever Answer:")
print(normal_chain.invoke(question))

print("\n Multi-Query Retriever Answer:")
print(multi_chain.invoke(question))