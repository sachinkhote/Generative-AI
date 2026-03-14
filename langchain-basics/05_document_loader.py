import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

print("Loading PDF...")
loader = PyPDFLoader("NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(f"first page review: \n {documents[0].page_content[:200]}...")

print("Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap= 50)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
print(f"first chunk: \n {chunks[0].page_content[:200]}")
print(f"Seconf chunk: \n {chunks[1].page_content}")

