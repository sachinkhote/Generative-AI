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
Week 3  ✅ — Advanced RAG Techniques
Week 4  ✅ — AI Agents
Week 5  🔄 — LangGraph + Agentic AI
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

## ✅ Week 3 — Advanced RAG Techniques

### Concepts Learned

**4 Advanced RAG Techniques:**

| Technique | What it does | Problem it solves |
|-----------|-------------|-------------------|
| Multi-Query Retrieval | LLM generates multiple search queries | Terminology mismatch |
| Conversation Memory | Rewrites questions using chat history | Follow-up questions fail |
| Reranking | Scores and reorders chunks by relevance | Wrong chunks ranked first |
| Hybrid Search | Combines semantic + keyword search | Missing exact term matches |

**Multi-Query Retrieval:**
```
User: "How many leaves do employees get?"
LLM generates:
→ "What is the leave entitlement for employees?"
→ "How many casual leave days are provided?"
→ "What is the vacation policy for staff?"
All 3 search ChromaDB → more relevant chunks found ✅
```

**Conversational RAG:**
```
Without contextualization:
"Can I carry them forward?" → searches as is → no match ❌

With contextualization:
"Can I carry them forward?" → rewritten using history →
"Can casual leaves be carried forward?" → correct match ✅
```

**Reranking:**
```
Retrieved 5 chunks → LLM scores each 1-10 → reorder by score
Most relevant chunk moves to position 1 → LLM sees it first ✅
```

**Hybrid Search:**
```
Semantic (ChromaDB) + Keyword (BM25) combined
weights=[0.5, 0.5] → equal importance
→ finds both meaning-based AND exact term matches ✅
```

### Files Created
```
advanced-rag/
├── 01_multi_query.py    — Multi-query retrieval comparison
├── 02_memory_rag.py     — Conversational RAG with memory
├── 03_reranking.py      — LLM-based chunk reranking
└── 04_hybrid_search.py  — BM25 + ChromaDB hybrid search
```

### Problems Faced & Solutions

| Problem | Cause | Solution |
|---------|-------|---------|
| Multi-query used generic terms | LLM generated Western HR terminology | Custom prompt with Indian HR terms |
| "Can I carry them forward?" failed | Retriever got vague pronoun query | Query contextualizer rewrites question |
| Repeated chunks in multi-query | Same chunks retrieved multiple times | Deduplication with `seen` set |
| chroma_db pushed to GitHub | Not in `.gitignore` | Added `**/chroma_db/` pattern |

### Key Learnings
- Chunk size matters more than retrieval technique — `size=1000` worked better than `size=500`
- Multi-query needs domain-specific prompt — generic LLM terms don't match Indian HR document
- Conversational RAG needs TWO prompts — one to rewrite query, one to answer
- Reranking adds extra LLM calls — slower but more accurate
- BM25 keyword search finds exact terms that semantic search misses
- Production systems use dedicated reranking models (Cohere, BGE) instead of LLM scoring

---

## ✅ Week 4 — AI Agents

### Concepts Learned

**What is an AI Agent?**
```
Normal Chatbot = Reactive (user asks → bot answers)
AI Agent       = Proactive (thinks → plans → uses tools → answers)

4 parts of every agent:
1. BRAIN    → LLM (thinks and decides)
2. TOOLS    → Functions the agent can call
3. MEMORY   → Remembers conversation
4. PLANNING → Decides which tool to use and when
```

**ReAct Loop in action:**
```
User asks → THOUGHT → ACTION (tool call) → OBSERVATION → repeat → FINAL ANSWER
```

**4 Types of Tools Built:**

| Tool | Purpose |
|------|---------|
| calculator | Math calculations — never wrong |
| TavilySearch | Real-time web search |
| hr_policy_search | RAG search on HR document |
| save_report | Save findings to a file |

**Key Agent Concepts:**
- `@tool` decorator converts Python function into agent tool
- Docstring of tool = description LLM reads to decide when to use it
- `agent_scratchpad` = agent's working memory during reasoning
- `verbose=True` = shows full Thought → Action → Observation loop
- `max_iterations` = prevents infinite loops

### Files Created
```
ai-agents/
├── 01_simple_agent.py      — Calculator + word counter agent
├── 02_web_search_agent.py  — Tavily web search + calculator
├── 03_rag_agent.py         — HR Policy RAG + web + calculator
└── 04_research_agent.py    — Full research agent with file saving
```

### Problems Faced & Solutions

| Problem | Cause | Solution |
|---------|-------|---------|
| AgentExecutor import error | Moved to langchain-classic | from langchain_classic.agents import AgentExecutor |
| Output showing raw list | Gemini returns list not string | Custom output cleaning loop |
| Agent picking wrong tool | Tool descriptions too vague | Made docstrings more specific |

### Key Learnings
- Tool docstring is the most important part — LLM reads it to decide when to use the tool
- Agent automatically chains multiple tools for complex questions
- verbose=True is essential for debugging agent behavior
- max_iterations=5 prevents agent from getting stuck in loops
- Same ReAct pattern from Prompt Engineering Week — now implemented in code!

---

## 🛠️ Tech Stack So Far

| Category | Tool | Used In |
|----------|------|---------|
| LLM | Gemini 1.5 Flash / 2.5 Flash | All projects |
| Framework | LangChain, LangChain-Classic | Week 2, 3, 4 |
| Vector DB | ChromaDB | Week 2, 3, 4 |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Week 2, 3, 4 |
| RAG Platform | Dify | HR Policy Assistant |
| Web Search | Tavily | Week 4 Agents |
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

### 4. AI Agents (4 progressively complex agents)
- **What:** Agents that use tools to answer questions accurately
- **Stack:** LangChain + Tavily + ChromaDB + Gemini
- **Concepts:** ReAct loop, Tool calling, Multi-tool chaining, File saving

---

## 💡 Interview Talking Points

**On AI Agents:**
> "An AI Agent combines an LLM with tools and a ReAct loop — Reasoning, Acting and Observing. The LLM reads tool descriptions to decide which tool to use, calls it, gets the result and continues reasoning until it has enough information to answer. I built agents that combined web search, RAG retrieval and calculations in a single response."

**On RAG:**
> "RAG stands for Retrieval Augmented Generation. Instead of relying on the LLM's training data, we first retrieve relevant chunks from our document, then pass them as context to the LLM to generate a grounded answer. This reduces hallucination and keeps answers accurate."

**On LangChain:**
> "LangChain is a framework that provides ready-made components for building GenAI applications — LLM connections, prompt templates, chains, memory and vector store integrations. It's model agnostic so you can switch between Gemini, OpenAI or Claude with minimal code changes."

**On Hybrid Search:**
> "Hybrid search combines semantic search using vector embeddings with keyword search using BM25. Semantic search finds meaning-based matches while keyword search finds exact term matches. Combining both gives more complete retrieval than either alone."

**On Reranking:**
> "After initial retrieval, reranking scores each chunk by relevance to the query and reorders them so the most relevant chunk is first. This improves answer quality because LLMs perform better when the most relevant context appears early."

---

*Last updated: Week 4 complete — Week 5 (LangGraph) starting next*