# 🧠 Advanced RAG: Theory, Architectures & Building a Better Chatbot

> **Reading Order:** `01_RAG_Practical_Guide.md` → this file → `agentmarkdown.md`
>
> This file covers **theory + database.py** (data layer). For the **agent layer** (`agent.py` + `chat_model.py`), see `agentmarkdown.md`.

## Table of Contents
1. [RAG Architecture Types](#1-rag-architectures)
2. [Advanced Retrieval Techniques](#2-advanced-retrieval)
3. [Hybrid Search (BM25 + Vector)](#3-hybrid-search)
4. [GraphRAG & Knowledge Graphs](#4-graphrag)
5. [Reducing Hallucination](#5-reducing-hallucination)
6. [Step-by-Step: Building the Improved Chatbot](#6-building-improved-chatbot)
7. [Complete Enhanced database.py Code](#7-complete-code)

---

## 1. RAG Architecture Types

### 1.1 — 2-Step RAG (Simplest)

```
User Question → Retrieve Documents → Generate Answer → Return
```

**How it works:** Always retrieves first, then generates. One LLM call.

**Pros:** Fast, predictable, easy to debug
**Cons:** Can't decide if retrieval is needed, no self-correction
**Best for:** FAQ bots, documentation search

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY this context:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4.1-mini")
)
answer = chain.invoke("What are the credit requirements?")
```

### 1.2 — Agentic RAG (Most Flexible)

```
User Question → Agent (LLM) → Decides: Need info? → Yes → Search tool → Enough? → Generate
                                                    → No → Answer directly
```

**How it works:** An LLM-powered agent **decides** when and what to retrieve. Can call tools multiple times.

**Pros:** Can handle complex multi-step questions, decides if retrieval is needed
**Cons:** More LLM calls = slower and costlier, harder to debug

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def search_handbook(query: str):
    """Search the IITM handbook for relevant information."""
    docs = vector_store.similarity_search(query, k=3)
    text = "\n\n".join(f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs)
    return text, docs

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[search_handbook],
    system_prompt="You help IITM students. Use search_handbook for any questions about rules, credits, or policies.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "How many credits to graduate?"}]})
```

### 1.3 — Hybrid RAG (Best for Production)

```
Question → Query Enhancement → Retrieve → Validate Relevance → Generate → Validate Answer → Return
                                    ↑                    |
                                    └── Refine Query ←──┘ (if insufficient)
```

**How it works:** Adds validation steps. Can refine queries and re-retrieve if results are poor.

**Pros:** Higher quality, self-correcting
**Cons:** More complex, multiple LLM calls
**Best for:** Production systems where answer quality matters

### Comparison Table

| Architecture | LLM Calls | Latency | Quality | Complexity | Your Chatbot |
|-------------|----------|---------|---------|------------|-------------|
| 2-Step | 1 | Fast ⚡ | Good | Low | Current approach |
| Agentic | Variable | Slow 🐢 | High | Medium | Future upgrade |
| **Hybrid** | **2-3** | **Medium** | **Best** | **Medium** | **Recommended** |

---

## 2. Advanced Retrieval Techniques

### 2.1 — Query Expansion / Multi-Query

**Problem:** A single query might miss relevant docs due to wording mismatch.
**Solution:** Generate multiple query variations, retrieve for each, merge results.

```python
from langchain_openai import ChatOpenAI

def expand_query(question: str, llm=None) -> list:
    """Generate 3 alternative phrasings of the question."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    prompt = f"""Generate 3 different ways to ask this question.
Return ONLY the questions, one per line. No numbering.

Original: {question}"""

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return [question] + queries  # Include original

def multi_query_retrieve(vector_store, question: str, k: int = 4) -> list:
    """Retrieve using multiple query variations, deduplicate results."""
    queries = expand_query(question)
    all_docs = []
    seen_contents = set()

    for query in queries:
        docs = vector_store.similarity_search(query, k=k)
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)

    return all_docs[:k * 2]  # Return top results
```

### 2.2 — HyDE (Hypothetical Document Embeddings)

**Problem:** Query embeddings and document embeddings live in different "spaces" — queries are short, docs are long.
**Solution:** Ask the LLM to generate a hypothetical answer, embed THAT, and search with it.

```python
def hyde_retrieve(vector_store, question: str, k: int = 4) -> list:
    """Use HyDE: generate hypothetical answer, then search with it."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

    prompt = f"""Write a short paragraph that would answer this question
based on an academic handbook. Write as if you found it in the document.

Question: {question}"""

    hypothetical = llm.invoke(prompt).content

    # Search using the hypothetical doc (closer to actual doc embeddings)
    return vector_store.similarity_search(hypothetical, k=k)
```

### 2.3 — Contextual Compression

**Problem:** Retrieved chunks contain irrelevant parts that dilute the context.
**Solution:** Extract only the relevant portions from each retrieved chunk.

```python
def compress_context(docs: list, question: str) -> str:
    """Extract only relevant parts from retrieved documents."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

    prompt = f"""Given this question: "{question}"

Extract ONLY the relevant sentences from each document below.
If a document has nothing relevant, skip it.

Documents:
"""
    for i, doc in enumerate(docs):
        prompt += f"\n--- Document {i+1} ---\n{doc.page_content}\n"

    prompt += "\nRelevant extracts:"
    response = llm.invoke(prompt)
    return response.content
```

### 2.4 — Reranking

**Problem:** Initial retrieval returns k results but ordering may not be optimal.
**Solution:** Use a cross-encoder or LLM to re-score and reorder results.

```python
def rerank_documents(docs: list, question: str, top_k: int = 4) -> list:
    """Use LLM to rerank retrieved documents by relevance."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

    scored_docs = []
    for doc in docs:
        prompt = f"""Rate how relevant this text is to the question on a scale of 0-10.
Return ONLY the number.

Question: {question}
Text: {doc.page_content[:500]}"""

        score = llm.invoke(prompt).content.strip()
        try:
            scored_docs.append((float(score), doc))
        except ValueError:
            scored_docs.append((0, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
```

---

## 3. Hybrid Search (BM25 + Vector)

### The Problem with Vector-Only Search

Vector search is great for **semantic** similarity but can miss **exact keyword** matches:

```
Query: "BSCS1002"  (a course code)
Vector search might return: "bachelor of science in computer science" (wrong!)
BM25 keyword search would match: "Course BSCS1002: Introduction to..." (correct!)
```

### Solution: Combine Both

> **Install required packages:** `pip install rank_bm25 langchain-classic`
>
> In LangChain v1, `EnsembleRetriever` moved to `langchain_classic.retrievers`.
> `BM25Retriever` remains in `langchain_community.retrievers` (requires `rank_bm25`).

```python
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(vector_store, documents, k=4):
    """Combine vector search (semantic) + BM25 (keyword) for best results."""

    # Vector retriever (semantic meaning)
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )

    # BM25 retriever (exact keyword matching)
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    # Combine with equal weights
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # 50% semantic, 50% keyword
    )
    return hybrid_retriever
```

### When to Use What

| Search Type | Good At | Bad At |
|-------------|---------|--------|
| Vector (semantic) | "What are the rules for..." | Exact codes, names, numbers |
| BM25 (keyword) | Course codes, specific terms | Understanding meaning |
| **Hybrid** | **Both!** | Slightly more setup |

---

## 4. GraphRAG & Knowledge Graphs

### What is GraphRAG?

Traditional RAG: Documents → Chunks → Flat vector search
GraphRAG: Documents → **Extract entities & relationships** → Graph database → Graph + vector search

```
Traditional: "How is course X related to Y?"
  → Finds chunks mentioning X OR Y separately
  → May miss the relationship

GraphRAG: Same question
  → Follows the graph edge: X --[prerequisite_of]--> Y
  → Directly finds the relationship!
```

### How GraphRAG Works

1. **Extract entities:** Courses, professors, departments, rules
2. **Extract relationships:** "prerequisite_of", "taught_by", "belongs_to"
3. **Build knowledge graph:** Nodes (entities) + Edges (relationships)
4. **Query:** Traverse the graph + vector search for comprehensive retrieval

### When GraphRAG Helps

| Scenario | Vector RAG | GraphRAG |
|----------|-----------|----------|
| "What is X?" | ✅ Great | ✅ Great |
| "How is X related to Y?" | ❌ Weak | ✅ Great |
| "What are all prerequisites for Z?" | ❌ Misses some | ✅ Complete |
| Multi-hop reasoning | ❌ Limited | ✅ Great |

### Simple Knowledge Graph Extraction

```python
def extract_entities_and_relations(text: str, llm=None) -> dict:
    """Extract entities and relationships from text using LLM."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

    prompt = f"""Extract entities and relationships from this text.

Text: {text}

Return as JSON:
{{
  "entities": [
    {{"name": "...", "type": "course/rule/department/person"}},
  ],
  "relationships": [
    {{"from": "...", "to": "...", "relation": "prerequisite_of/part_of/requires"}},
  ]
}}"""

    response = llm.invoke(prompt)
    try:
        import json
        return json.loads(response.content)
    except:
        return {"entities": [], "relationships": []}
```

### Note on GraphRAG for Your Chatbot

For the IITM handbook, GraphRAG would be valuable for:
- Course prerequisite chains
- Credit requirement relationships
- Department → Course → Professor mappings

However, for a first version, **Hybrid RAG (BM25 + Vector + Query Expansion)** gives you 80% of the benefit with 20% of the complexity. GraphRAG is a future upgrade.

---

## 5. Reducing Hallucination

### Why Chatbots Hallucinate

1. **Insufficient context:** Retrieved chunks don't contain the answer
2. **Conflicting context:** Multiple chunks say different things
3. **LLM fills gaps:** Model invents plausible-sounding but wrong info
4. **Poor prompt design:** Prompt doesn't tell LLM to say "I don't know"

### Anti-Hallucination Strategies

#### Strategy 1: Strict System Prompt

```python
SYSTEM_PROMPT = """You are an IITM student assistant. Answer ONLY based on the context below.

RULES:
1. If the answer is NOT in the context, say: "I couldn't find this in the handbook."
2. NEVER make up information. If unsure, say so.
3. Quote specific sections when possible.
4. If the context is partially relevant, state what you found and what's missing.

CONTEXT:
{context}
"""
```

#### Strategy 2: Source Attribution

```python
def format_docs_with_sources(docs):
    """Format docs with source info so LLM can cite them."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)
```

#### Strategy 3: Confidence Check

```python
def answer_with_confidence(question, context, llm):
    """Generate answer with confidence score."""
    prompt = f"""Based on this context, answer the question.

Context: {context}

Question: {question}

Respond in this format:
CONFIDENCE: [HIGH/MEDIUM/LOW]
ANSWER: [your answer]
SOURCES_USED: [which source numbers you used]

If you cannot find the answer, set CONFIDENCE to LOW and say you couldn't find it."""

    return llm.invoke(prompt)
```

#### Strategy 4: Retrieval Validation

```python
def validate_retrieval(docs, question, llm):
    """Check if retrieved docs actually answer the question."""
    context = "\n".join(d.page_content[:200] for d in docs)

    prompt = f"""Do these documents contain information to answer: "{question}"?
Documents preview: {context}
Answer YES or NO only."""

    response = llm.invoke(prompt)
    return "YES" in response.content.upper()
```

---

## 6. Building the Improved Chatbot — Step by Step

### Architecture Overview

```
User Question
    │
    ▼
┌─────────────────┐
│ Query Enhancement│ ← Multi-query expansion
└────────┬────────┘
         ▼
┌─────────────────┐
│ Hybrid Retrieval │ ← MMR + Metadata filtering
└────────┬────────┘
         ▼
┌─────────────────┐
│ Context Building │ ← Format with sources
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Generation  │ ← Anti-hallucination prompt
└────────┬────────┘
         ▼
    Grounded Answer
```

### Step 1: Enhanced Document Loading

Already covered in Markdown 1. Key improvements:
- Multi-format support (PDF, TXT, HTML, URLs)
- Metadata enrichment (file_type, source)
- Scan entire `data/` directory

### Step 2: Better Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def smart_split(documents):
    """Split with parameters tuned for academic handbooks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata with chunk index
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
```

### Step 3: Vector Store with Dedup

```python
import hashlib
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vector_store():
    return Chroma(
        collection_name="student_docs",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="chroma_db",
    )

def doc_id(doc):
    s = f"{doc.metadata.get('source','')}:{doc.page_content[:500]}"
    return hashlib.md5(s.encode()).hexdigest()

def add_with_dedup(vs, chunks):
    ids = [doc_id(c) for c in chunks]
    existing = set(vs.get()["ids"] or [])
    new = [(c, i) for c, i in zip(chunks, ids) if i not in existing]
    if new:
        vs.add_documents([c for c,_ in new], ids=[i for _,i in new])
        print(f"✅ Added {len(new)} new chunks")
```

### Step 4: Smart Retriever

```python
def get_smart_retriever(vs, k=5):
    """MMR retriever for relevant + diverse results."""
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
```

### Step 5: Agent Layer (see `agentmarkdown.md`)

The **agent layer** (`agent.py` + `chat_model.py`) is covered in `agentmarkdown.md`. It provides:

- **Version 1 (Basic):** 2-step RAG chain using `create_agent` + `dynamic_prompt` middleware
- **Version 2 (Agentic):** Full agentic RAG where the LLM decides when to search

Both versions import `get_vector_store()` from the `database.py` built below.

### Step 6: Wiring It All Together

The final project structure:

```
project/
├── data/                  # Your documents (PDF, TXT, HTML, resources.txt)
├── chroma_db/             # Persistent vector store (auto-created)
├── database.py            # Data layer — load, chunk, store, retrieve
├── chat_model.py          # LLM initialization
├── agent.py               # Agent layer — RAG chain or agentic RAG
└── main.py                # Entry point
```

**Workflow:**
```bash
# Step 1: Build the vector store (run once, or when new docs added)
python database.py

# Step 2: Ask questions
python agent.py
```

---

## 7. Complete Enhanced database.py

This is the final production-ready `database.py` — a **pure data module** (load, chunk, store, retrieve). The agent/LLM layer lives in `agent.py` (see `agentmarkdown.md`).

```python
"""
database.py — Enhanced Vector Store for IITM Handbook RAG Chatbot.

This module handles the DATA LAYER only:
- Multi-format document loading (PDF, TXT, HTML, URLs)
- Smart chunking with metadata enrichment
- Deduplication via content hashing
- MMR retrieval for diverse results

The AGENT LAYER (LLM, prompts, chains) lives in agent.py.
See agentmarkdown.md for the agent code.
"""

import os
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader, WebBaseLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "student_docs"
DATA_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


# ════════════════════════════════════════════
# EMBEDDINGS
# ════════════════════════════════════════════
def get_embeddings():
    """Create embedding model. Same model must be used for indexing AND querying."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ════════════════════════════════════════════
# DOCUMENT LOADERS
# ════════════════════════════════════════════
def _load_pdf(path):
    return PyPDFLoader(path).load()

def _load_text(path):
    return TextLoader(path, encoding="utf-8").load()

def _load_html(path):
    return UnstructuredHTMLLoader(path).load()

def _load_urls(path):
    docs = []
    with open(path) as f:
        urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for url in urls:
        try:
            loaded = WebBaseLoader(web_paths=[url]).load()
            for d in loaded:
                d.metadata["file_type"] = "web"
            docs.extend(loaded)
            print(f"  ✅ {url}")
        except Exception as e:
            print(f"  ❌ {url} → {e}")
    return docs

_LOADER_MAP = {".pdf": _load_pdf, ".txt": _load_text, ".html": _load_html, ".htm": _load_html}

def load_all_documents(data_dir=DATA_DIR):
    """Load all supported files from data directory."""
    all_docs = []
    for name in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, name)
        if name == "resources.txt":
            print(f"🌐 URLs from {name}...")
            all_docs.extend(_load_urls(fpath))
        elif os.path.isfile(fpath):
            ext = os.path.splitext(name)[1].lower()
            fn = _LOADER_MAP.get(ext)
            if fn:
                print(f"📄 {name}...")
                loaded = fn(fpath)
                for d in loaded:
                    d.metadata["file_type"] = ext.lstrip(".")
                all_docs.extend(loaded)
            else:
                print(f"⚠️  Skip {name} (unsupported)")
    print(f"📊 Loaded {len(all_docs)} documents")
    return all_docs


# ════════════════════════════════════════════
# CHUNKING
# ════════════════════════════════════════════
def split_documents(documents):
    """Split with parameters tuned for academic handbooks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    print(f"✂️  {len(chunks)} chunks")
    return chunks


# ════════════════════════════════════════════
# VECTOR STORE
# ════════════════════════════════════════════
def _doc_id(doc):
    """Generate unique ID from content + source for deduplication."""
    s = f"{doc.metadata.get('source','')}:{doc.page_content[:500]}"
    return hashlib.md5(s.encode()).hexdigest()

def get_vector_store():
    """Get or create persistent ChromaDB vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

def add_documents_to_store(vs, chunks):
    """Add chunks with deduplication — safe to run multiple times."""
    ids = [_doc_id(c) for c in chunks]
    existing = set(vs.get()["ids"] or [])
    new = [(c, i) for c, i in zip(chunks, ids) if i not in existing]
    if new:
        vs.add_documents([c for c,_ in new], ids=[i for _,i in new])
        print(f"✅ Added {len(new)} (skipped {len(chunks)-len(new)})")
    else:
        print("ℹ️  All docs already exist")

def get_store_stats(vs):
    """Get stats about the vector store."""
    data = vs.get()
    sources = {m.get("source","?") for m in data["metadatas"]}
    return {"total": len(data["ids"]), "sources": sorted(sources)}


# ════════════════════════════════════════════
# RETRIEVER
# ════════════════════════════════════════════
def get_retriever(vs, search_type="mmr", k=5):
    """Create retriever. Default: MMR for diverse results."""
    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs["fetch_k"] = k * 3
    return vs.as_retriever(search_type=search_type, search_kwargs=kwargs)


# ════════════════════════════════════════════
# BUILD PIPELINE
# ════════════════════════════════════════════
def build_vector_store(data_dir=DATA_DIR):
    """Full pipeline: Load → Split → Embed → Store."""
    print("🚀 Building Vector Store...")
    docs = load_all_documents(data_dir)
    if not docs:
        print("❌ No documents found!")
        return get_vector_store()
    chunks = split_documents(docs)
    vs = get_vector_store()
    add_documents_to_store(vs, chunks)
    stats = get_store_stats(vs)
    print(f"📈 {stats['total']} chunks from {len(stats['sources'])} sources")
    return vs


if __name__ == "__main__":
    build_vector_store()
```

---

## Summary: Making Your Chatbot Much Better

### Quick Wins (Do Now)
1. ✅ Multi-format loading (not just PDF)
2. ✅ Deduplication with content hashing
3. ✅ MMR retrieval instead of plain similarity
4. ✅ Anti-hallucination system prompt
5. ✅ Source attribution in context

### Medium Effort (Do Next)
6. Query expansion for better recall
7. Contextual compression to reduce noise
8. Hybrid search (BM25 + vector) for exact matches

### Advanced (Future)
9. Reranking with cross-encoders
10. GraphRAG for relationship queries
11. Self-correcting retrieval loops (Hybrid RAG architecture)
