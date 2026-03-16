import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_rerank"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Reranker using LLM
def rerank_docs(question, docs):
    scored_docs = []
    for doc in docs:
        rerank_prompt = f"""Rate how relevant this chunk is to answer the question.
Return ONLY a number from 1-10. Nothing else.

Question: {question}
Chunk: {doc.page_content[:300]}"""

        try:
            score_response = llm.invoke(rerank_prompt)
            score = float(score_response.content.strip())
        except Exception as e:
            score = 0

        scored_docs.append((score, doc))

    # Sort by score — highest first
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    print("\n📊 Chunk Relevance Scores:")
    for score, doc in scored_docs:
        print(f"  Score {score}: {doc.page_content[:80]}...")

    return [doc for score, doc in scored_docs]

# RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR Policy Assistant.
Answer ONLY based on the provided context.
If not found say: 'I couldn't find this in the HR policy.'"""),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("✅ Ready!")
print("="*50)
print("   RAG with Reranking 🔄")
print("="*50)

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    if question.strip() == "":
        continue

    # Step 1 — Retrieve chunks
    docs = retriever.invoke(question)
    print(f"\n📥 Retrieved {len(docs)} chunks")

    # Step 2 — Rerank chunks
    print("🔄 Reranking...")
    reranked_docs = rerank_docs(question, docs)

    # Step 3 — Use top 3 reranked chunks
    context = format_docs(reranked_docs[:3])

    # Step 4 — Generate answer
    chain = prompt | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\n🤖 Assistant: {answer}")
    print("-"*50)