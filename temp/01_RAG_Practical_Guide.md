# 📚 Practical RAG Guide: From Zero to Production

> **Reading Order:** this file → `02_Advanced_RAG_Theory_and_Better_Chatbot.md` → `agentmarkdown.md`
>
> This file covers **fundamentals + basic database.py**. For advanced theory see Markdown 2. For the agent layer see `agentmarkdown.md`.

## Table of Contents
1. [What is RAG?](#1-what-is-rag)
2. [Analysis of Your Current database.py](#2-analysis-of-current-code)
3. [RAG Pipeline Deep Dive](#3-rag-pipeline)
4. [Document Loading](#4-document-loading)
5. [Chunking Strategies](#5-chunking)
6. [Embeddings](#6-embeddings)
7. [Vector Store (ChromaDB)](#7-vector-store)
8. [Retrieval Strategies](#8-retrieval)
9. [Complete Rebuilt database.py](#9-rebuilt-code)
10. [Interview Questions](#10-interview-questions)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** = Retrieve relevant docs → Feed them to LLM → Get grounded answers.

```
User Query → [Retriever] → Relevant Chunks → [LLM + Context] → Answer
```

### Why RAG?
- LLMs have **finite context** — can't read your entire document corpus
- LLMs have **static knowledge** — training data is frozen
- RAG gives LLMs **access to your specific data** at query time

### The 5-Step Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   1. LOAD   │───▶│  2. SPLIT   │───▶│  3. EMBED   │───▶│  4. STORE   │───▶│ 5. RETRIEVE │
│  Documents  │    │  into chunks│    │  to vectors │    │  in VectorDB│    │  & Generate │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Step 1 - LOAD:** Read raw files into LangChain `Document` objects. Each Document has `page_content` (text), `metadata` (dict like `{"source": "file.pdf", "page": 0}`), and optional `id`.

**Step 2 - SPLIT:** Break documents into smaller chunks. Why? Embedding models have token limits, smaller chunks = more precise retrieval, and it reduces noise.

**Step 3 - EMBED:** Convert text chunks into number vectors. Similar text → similar vectors (close in vector space). Uses models like `text-embedding-3-small`.

**Step 4 - STORE:** Save vectors in a database (ChromaDB). Enables fast similarity search and persists to disk.

**Step 5 - RETRIEVE:** Query → embed → find nearest vectors → return relevant documents → feed to LLM.

---

## 2. Analysis of Current database.py

### Issues Found

| # | Issue | Why It's Bad | Fix |
|---|-------|-------------|-----|
| 1 | Only loads PDF files | Can't handle .txt, .html, URLs | Add multi-format loader |
| 2 | Hardcoded single file path | Can't process multiple files | Scan entire `data/` directory |
| 3 | `Chroma.from_documents()` used | Recreates DB each time, no dedup | Use `add_documents()` with IDs |
| 4 | No duplicate detection | Re-running adds same docs again | Hash-based document IDs |
| 5 | Only `similarity` search | Misses diverse results | Add MMR search option |
| 6 | `chunk_size=800` is small | Loses context in academic docs | Use 1000-1500 for handbooks |
| 7 | `chunk_overlap=100` is small | Context breaks at boundaries | Use 200+ overlap |
| 8 | No metadata enrichment | Can't filter by source/type | Add file_type, source metadata |
| 9 | No `add_start_index` | Can't trace chunk position | Enable start_index tracking |
| 10 | Embeddings re-created each call | Wasteful, no centralization | Single config function |

---

## 3. Document Loading

### Loading PDFs

```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str) -> list:
    """Load a PDF file. Returns one Document per page."""
    loader = PyPDFLoader(file_path)
    return loader.load()
```

### Loading Text Files

```python
from langchain_community.document_loaders import TextLoader

def load_text(file_path: str) -> list:
    """Load a plain text file as a single Document."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
```

### Loading HTML Files

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

def load_html(file_path: str) -> list:
    """Load an HTML file, extracting text content."""
    loader = UnstructuredHTMLLoader(file_path)
    return loader.load()
```

### Loading Web URLs (from resources.txt)

```python
from langchain_community.document_loaders import WebBaseLoader

def load_urls_from_file(resources_path: str) -> list:
    """Read URLs from resources.txt and load each webpage."""
    all_docs = []
    with open(resources_path, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url])
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  ✅ Loaded: {url}")
        except Exception as e:
            print(f"  ❌ Failed: {url} → {e}")
    return all_docs
```

### Universal Loader (Auto-detect)

```python
import os

def load_file(file_path: str) -> list:
    """Auto-detect file type and load accordingly."""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".html": load_html,
        ".htm": load_html,
    }
    loader_fn = loaders.get(ext)
    if loader_fn is None:
        print(f"  ⚠️  Unsupported file type: {ext}")
        return []
    return loader_fn(file_path)
```

### Load Entire Data Directory

```python
def load_all_documents(data_dir: str = "data") -> list:
    """Load all supported files from the data directory."""
    all_docs = []
    for filename in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, filename)
        if filename == "resources.txt":
            print(f"🌐 Loading URLs from {filename}...")
            docs = load_urls_from_file(file_path)
            all_docs.extend(docs)
        elif os.path.isfile(file_path):
            print(f"📄 Loading {filename}...")
            docs = load_file(file_path)
            # Add file_type metadata
            ext = os.path.splitext(filename)[1].lstrip(".")
            for doc in docs:
                doc.metadata["file_type"] = ext
            all_docs.extend(docs)
    print(f"\n📊 Total documents loaded: {len(all_docs)}")
    return all_docs
```

---

## 4. Chunking Strategies

### Key Parameters

| Parameter | What It Does | Recommended |
|-----------|-------------|-------------|
| `chunk_size` | Max characters per chunk | 1000–1500 for academic docs |
| `chunk_overlap` | Shared chars between chunks | 200 (20% of chunk_size) |
| `separators` | Split priority order | `["\n\n", "\n", ". ", " "]` |
| `add_start_index` | Track position in original | Always `True` |

### RecursiveCharacterTextSplitter (Recommended)

Tries to split on natural boundaries in this order:
1. `\n\n` (paragraphs) → 2. `\n` (lines) → 3. `. ` (sentences) → 4. ` ` (words) → 5. characters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_text_splitter():
    """Create splitter optimized for academic/handbook documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def split_documents(documents: list) -> list:
    """Split documents into chunks."""
    splitter = create_text_splitter()
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks
```

### Why chunk_overlap Matters

```
WITHOUT overlap:
  Chunk 1: "...student must complete 8 courses"
  Chunk 2: "each worth 4 credits to graduate."
  → Query "credit requirements" might miss the full answer!

WITH overlap (200):
  Chunk 1: "...student must complete 8 courses each worth 4 credits"
  Chunk 2: "each worth 4 credits to graduate. Additionally..."
  → Both chunks capture the complete context!
```

---

## 5. Embeddings

### What are Embeddings?

Converting text to number arrays so similar meanings produce similar vectors:

```
"credit requirements"  →  [0.12, -0.45, 0.78, ...]  (1536 dims)
"how many credits"     →  [0.11, -0.44, 0.77, ...]  (very similar!)
"weather forecast"     →  [0.89, 0.23, -0.56, ...]  (very different!)
```

```python
from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    """Create embedding model. Same model for indexing AND querying."""
    return OpenAIEmbeddings(model="text-embedding-3-small")
```

| Model | Dimensions | Cost | Notes |
|-------|-----------|------|-------|
| `text-embedding-3-small` | 1536 | Cheapest | Good for most uses |
| `text-embedding-3-large` | 3072 | 6x more | Better accuracy |

> **Critical Rule:** Once you embed docs with a model, you MUST use the same model for queries.

---

## 6. Vector Store (ChromaDB) — Latest API

### Initialization

```python
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "student_docs"

def get_vector_store() -> Chroma:
    """Get or create a persistent ChromaDB vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
```

### Adding Documents with Deduplication

```python
import hashlib

def generate_doc_id(doc) -> str:
    """Generate unique ID from content + source."""
    content = doc.page_content
    source = doc.metadata.get("source", "")
    unique_string = f"{source}:{content[:500]}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def add_documents_to_store(vector_store, chunks):
    """Add chunks with dedup — safe to run multiple times."""
    ids = [generate_doc_id(c) for c in chunks]
    existing = vector_store.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    new_chunks, new_ids = [], []
    for chunk, doc_id in zip(chunks, ids):
        if doc_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(doc_id)

    if new_chunks:
        vector_store.add_documents(documents=new_chunks, ids=new_ids)
        print(f"✅ Added {len(new_chunks)} new (skipped {len(chunks)-len(new_chunks)} dupes)")
    else:
        print("ℹ️  No new documents to add")
```

---

## 7. Retrieval Strategies

### Similarity Search (Default)

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
```

### MMR — Maximal Marginal Relevance (Best Default)

Balances **relevance** + **diversity**. Avoids returning near-duplicate chunks.

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 12},
)
```

How MMR works: Fetch 12 similar → pick most relevant → for each next pick, choose most relevant BUT most different from already selected → repeat until k=4.

### Similarity with Score Threshold

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5},
)
```

### Metadata Filtering

```python
results = vector_store.similarity_search(
    "credit requirements", k=4,
    filter={"source": "data/Handbook.pdf"},
)
```

### Which to Use?

| Strategy | Best For | Your Chatbot |
|----------|---------|--------------|
| Similarity | Specific lookups | Direct questions |
| **MMR** | **Broad questions** | **Best default for handbook** |
| Threshold | High-precision needs | Strict factual Q&A |
| Filter | Multi-source | Filter by file when needed |

---

## 8. Complete Rebuilt database.py

```python
"""
database.py — Vector store management for IITM Handbook RAG Chatbot.
Handles: loading (PDF/TXT/HTML/URLs), chunking, embedding, ChromaDB storage, retrieval.
"""
import os
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader, WebBaseLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ── CONFIG ──
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "student_docs"
DATA_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── 1. EMBEDDINGS ──
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ── 2. LOADERS ──
def load_pdf(path):
    return PyPDFLoader(path).load()

def load_text(path):
    return TextLoader(path, encoding="utf-8").load()

def load_html(path):
    return UnstructuredHTMLLoader(path).load()

def load_urls_from_file(path):
    all_docs = []
    with open(path, "r") as f:
        urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for url in urls:
        try:
            docs = WebBaseLoader(web_paths=[url]).load()
            for d in docs:
                d.metadata["file_type"] = "web"
            all_docs.extend(docs)
            print(f"  ✅ Loaded: {url}")
        except Exception as e:
            print(f"  ❌ Failed: {url} → {e}")
    return all_docs

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    loader_map = {".pdf": load_pdf, ".txt": load_text, ".html": load_html, ".htm": load_html}
    fn = loader_map.get(ext)
    if not fn:
        print(f"  ⚠️  Unsupported: {ext}")
        return []
    docs = fn(path)
    for d in docs:
        d.metadata["file_type"] = ext.lstrip(".")
    return docs

def load_all_documents(data_dir=DATA_DIR):
    all_docs = []
    for name in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, name)
        if name == "resources.txt":
            print(f"🌐 Loading URLs from {name}...")
            all_docs.extend(load_urls_from_file(fpath))
        elif os.path.isfile(fpath):
            print(f"📄 Loading {name}...")
            all_docs.extend(load_file(fpath))
    print(f"\n📊 Total documents loaded: {len(all_docs)}")
    return all_docs

# ── 3. CHUNKING ──
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    return chunks

# ── 4. VECTOR STORE ──
def _doc_id(doc):
    s = f"{doc.metadata.get('source','')}:{doc.page_content[:500]}"
    return hashlib.md5(s.encode()).hexdigest()

def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

def add_documents_to_store(vs, chunks):
    ids = [_doc_id(c) for c in chunks]
    existing_ids = set(vs.get()["ids"] or [])
    new = [(c, i) for c, i in zip(chunks, ids) if i not in existing_ids]
    if new:
        vs.add_documents(documents=[c for c,_ in new], ids=[i for _,i in new])
        print(f"✅ Added {len(new)} new (skipped {len(chunks)-len(new)} dupes)")
    else:
        print("ℹ️  No new documents to add")

# ── 5. RETRIEVER ──
def get_retriever(vs, search_type="mmr", k=4):
    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs["fetch_k"] = k * 3
    if search_type == "similarity_score_threshold":
        kwargs["score_threshold"] = 0.5
    return vs.as_retriever(search_type=search_type, search_kwargs=kwargs)

# ── 6. BUILD PIPELINE ──
def build_vector_store(data_dir=DATA_DIR):
    print("🚀 Building Vector Store...")
    documents = load_all_documents(data_dir)
    if not documents:
        print("❌ No documents found!")
        return get_vector_store()
    chunks = split_documents(documents)
    vs = get_vector_store()
    add_documents_to_store(vs, chunks)
    stats = vs.get()
    print(f"📈 Total chunks in store: {len(stats['ids'])}")
    return vs

def load_vector_store():
    """Backwards compatible: load existing store."""
    return get_vector_store()

if __name__ == "__main__":
    build_vector_store()
```

---

## 9. Interview Questions

**Q: What is RAG?**
Retrieval-Augmented Generation — fetch relevant docs from a knowledge base and include them as context for the LLM.

**Q: Why chunk documents?**
Embedding models have token limits; smaller chunks = more precise retrieval; reduces noise in LLM context.

**Q: What is chunk_overlap?**
Characters shared between adjacent chunks. Prevents losing context that spans boundaries.

**Q: Similarity search vs MMR?**
Similarity returns k most similar (may be redundant). MMR balances relevance with diversity by iteratively selecting documents different from already-selected ones.

**Q: How to prevent duplicates in ChromaDB?**
Hash-based IDs from content+source. Check existing IDs before adding.

**Q: What is RecursiveCharacterTextSplitter?**
Splits text by trying paragraph breaks first, then lines, sentences, words — keeping natural boundaries.

**Q: Can you change embedding models after indexing?**
No. You must re-embed all documents if you switch models.

**Q: What is `add_start_index=True`?**
Preserves the character position where each chunk starts in the original document — useful for tracing back to source.

**Q: What does `fetch_k` do in MMR?**
It's how many candidates to initially fetch before selecting the diverse subset of `k` results.

**Q: How would you handle new files added to the data directory?**
Re-run `build_vector_store()`. Hash-based dedup ensures only new content gets embedded and stored.
