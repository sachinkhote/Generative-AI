# 🤖 Generative AI — Learning Journey
### From Data Analytics → GenAI Developer
**Started:** March 2026 | **Target:** Job-ready in 8 weeks

---

## 👨‍💻 About Me
- **Background:** Data Analyst fresher | MCA (ML/AI)
- **Already knew:** Python, Pandas, SQL, Machine Learning basics
- **Goal:** Become a GenAI Developer specializing in RAG, LangChain and Agentic AI

---

## 🗺️ Learning Roadmap

```
Week 1  ✅ — LLM Foundations + Prompt Engineering
Week 2  ✅ — LangChain Basics + RAG in Code
Week 3  🔄 — Advanced RAG Techniques
Week 4  ⬜ — Advanced RAG + Conversation Memory
Week 5  ⬜ — AI Agents
Week 6  ⬜ — Agentic AI + Automation
Week 7  ⬜ — Portfolio Projects
Week 8  ⬜ — Interview Preparation
```

---

## ✅ Week 1 — LLM Foundations + Prompt Engineering

### Concepts Learned

**How LLMs Work:**
- LLMs are next-word predictors trained on billions of text documents
- Text is broken into **tokens** (not words) — ~3-4 characters per token
- **Context Window** — maximum tokens LLM can process at once
- **Temperature** — controls creativity of output
  - `0.0` = deterministic, factual
  - `0.7` = balanced, natural
  - `1.5+` = creative, unpredictable

**6 Prompt Engineering Techniques:**

| Technique | What it does | When to use |
|-----------|-------------|-------------|
| Zero-shot | Ask directly with Role+Task+Format+Constraint | Simple clear tasks |
| Few-shot | Give 2-3 examples to teach the pattern | Need specific output format |
| Chain of Thought | "Think step by step" for complex reasoning | Multi-step problems |
| System/User/Assistant | System sets rules, User asks, LLM replies | Every production app |
| ReAct | Thought → Action → Observation loop | AI Agents |
| Output Formatting | Get JSON, bullets, tables | Developer use cases |

### Project Built
**Job Description Extractor** — `job-description-extractor/`
- Takes any job description as input
- Extracts job title, experience, skills, salary as clean JSON
- Uses Few-shot + Output Formatting techniques
- Built with: Python + Gemini API

### Key Learnings
- Temperature `0` gives same answer every time — best for facts
- System prompt is the most powerful tool to control LLM behavior
- Few-shot examples must be **consistent** — inconsistent examples confuse the LLM
- Output formatting is critical for developer use — always specify exact format

---

## ✅ Week 2 — LangChain Basics + RAG in Code

### Concepts Learned

**Why LangChain?**
- Before LangChain — every developer wrote the same boilerplate code
- LangChain provides ready-made components for GenAI apps
- **Model agnostic** — switch from Gemini to OpenAI in one line

**6 LangChain Components:**

| Component | What it does |
|-----------|-------------|
| LLM Connection | Connect to any model — Gemini, OpenAI, Claude |
| Prompt Templates | Reusable prompts with variables |
| Chains (`\|`) | Connect steps — output of one becomes input of next |
| Memory | Store conversation history for multi-turn chat |
| Document Loaders | Load PDFs, CSVs, websites in Python |
| Vector Stores | Store and search embeddings locally |

**RAG Pipeline Built from Scratch:**
```
PDF (34 pages)
      │
PyPDFLoader → load pages
      │
RecursiveCharacterTextSplitter → 159 chunks (size=500, overlap=50)
      │
HuggingFaceEmbeddings (all-MiniLM-L6-v2) → convert to vectors
      │
ChromaDB → store vectors locally
      │
Retriever (k=3) → find top 3 relevant chunks
      │
ChatPromptTemplate → build prompt with context
      │
Gemini LLM → generate grounded answer
      │
StrOutputParser → clean text output ✅
```

**LCEL — LangChain Expression Language:**
```python
chain = prompt | llm | parser
# Output of prompt → input of llm → output → input of parser
```

### Files Created
```
langchain-basics/
├── 01_llm_connect.py       — Connect to Gemini via LangChain
├── 02_prompt_templates.py  — Reusable prompt templates
├── 03_chains.py            — LCEL chains with | operator
├── 04_memory.py            — Conversation memory
├── 05_document_loader.py   — Load and chunk PDF
└── 06_rag_langchain.py     — Full RAG pipeline
```

### Problems Faced & Solutions

| Problem | Cause | Solution |
|---------|-------|---------|
| "leaves" query returned wrong chunks | Keyword retrieval is terminology sensitive | Use exact document terminology |
| ECO-INVERTED index missed casual leave | Basic keyword matching, not semantic | High Quality mode or rephrase query |
| Gemini embedding 404 error | Dify used wrong model name internally | Switched back to Economical mode |
| `.env` pushed to GitHub accidentally | `.gitignore` not set up correctly | `git rm --cached .env` + fix `.gitignore` |

### Key Learnings
- **Retrieval quality > LLM quality** in RAG — "garbage in, garbage out"
- `chunk_overlap` prevents important info from being lost at chunk boundaries
- ChromaDB saves vectors locally — no cloud needed for development
- LangChain's `|` operator makes complex pipelines readable and clean
- Document terminology must match query terminology for keyword retrieval

---

## 🛠️ Tech Stack So Far

| Category | Tool | Used In |
|----------|------|---------|
| LLM | Gemini 1.5 Flash / 2.5 Flash | All projects |
| Framework | LangChain | Week 2 |
| Vector DB | ChromaDB | Week 2 RAG |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Week 2 RAG |
| RAG Platform | Dify | HR Policy Assistant |
| Language | Python | All projects |

---

## 📁 Projects Built

### 1. HR Policy Assistant
- **What:** RAG chatbot answering HR policy questions from PDF
- **Stack:** Python + Dify + Gemini API
- **Repo:** [HR-Policy-Assistant](https://github.com/sachinkhote/HR-Policy-Assistant)
- **Concepts:** RAG, Knowledge Base, System Prompts, API calls

### 2. Job Description Extractor
- **What:** Extracts structured info from job descriptions as JSON
- **Stack:** Python + Gemini API
- **Concepts:** Few-shot prompting, Output formatting, JSON extraction

### 3. LangChain RAG Pipeline
- **What:** HR Policy Assistant rebuilt in pure LangChain code
- **Stack:** LangChain + ChromaDB + HuggingFace + Gemini
- **Concepts:** Document loading, Chunking, Embeddings, Vector search, RAG chain

---

## 💡 Interview Talking Points

**On RAG:**
> "RAG stands for Retrieval Augmented Generation. Instead of relying on the LLM's training data, we first retrieve relevant chunks from our document, then pass them as context to the LLM to generate a grounded answer. This reduces hallucination and keeps answers accurate."

**On LangChain:**
> "LangChain is a framework that provides ready-made components for building GenAI applications — LLM connections, prompt templates, chains, memory and vector store integrations. It's model agnostic so you can switch between Gemini, OpenAI or Claude with minimal code changes."

**On the "leaves vs leave" problem:**
> "I observed that RAG retrieval accuracy heavily depends on query-document terminology alignment when using keyword-based retrieval. This led me to explore semantic embeddings and multi-query retrieval techniques to solve the problem."

---

*Last updated: Week 2 complete — Week 3 starting next*