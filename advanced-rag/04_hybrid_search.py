import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load and chunk PDF
print("📄 Loading PDF...")
loader = PyPDFLoader("NHPS-POLICIES-Revised-1.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} chunks created")

# Semantic retriever (ChromaDB)
print("🗄️ Creating semantic retriever...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_hybrid"
)
semantic_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Keyword retriever (BM25)
print("🔤 Creating keyword retriever...")
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 3

# Hybrid retriever — combines both!
print("🔀 Creating hybrid retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.5, 0.5]  # 50% semantic + 50% keyword
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
    seen = set()
    unique = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return "\n\n".join(doc.page_content for doc in unique)

# Three chains for comparison
semantic_chain = (
    {"context": semantic_retriever | format_docs, "question": lambda x: x}
    | prompt | llm | parser
)

keyword_chain = (
    {"context": keyword_retriever | format_docs, "question": lambda x: x}
    | prompt | llm | parser
)

hybrid_chain = (
    {"context": hybrid_retriever | format_docs, "question": lambda x: x}
    | prompt | llm | parser
)

print("\n" + "="*50)
print("   Hybrid Search Comparison 🔀")
print("="*50)

questions = [
    "What is the CL encashment policy?",
    "What happens on unauthorized absence?"
]

for question in questions:
    print(f"\n❓ Question: {question}")
    print("\n📌 Semantic Search Answer:")
    print(semantic_chain.invoke(question))
    print("\n🔤 Keyword Search Answer:")
    print(keyword_chain.invoke(question))
    print("\n🔀 Hybrid Search Answer:")
    print(hybrid_chain.invoke(question))
    print("\n" + "="*50)