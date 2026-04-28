import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# Where ChromaDB will persist the vectors on disk
CHROMA_DIR = "chroma_db"


def build_vector_store(pdf_path: str) -> Chroma:
    """
    Load a PDF, split it into chunks, embed them, and store in ChromaDB.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Chroma: A ready-to-query vector store.
    """

    # ── 1. LOAD ────────────────────────────────────────────
    # PyPDFLoader reads each page of the PDF as a LangChain "Document".
    # A Document has two fields:
    #   .page_content  → the raw text of that page
    #   .metadata      → dict with extras like {"source": "file.pdf", "page": 0}
    print(f"📄  Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"    Loaded {len(pages)} page(s)")

    # ── 2. SPLIT ───────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — keeping context intact.
    #
    #   chunk_size    = max characters per chunk  (~150 words)
    #   chunk_overlap = characters shared between adjacent chunks
    #                   (so we don't lose context at the boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(pages)
    print(f"    Split into {len(chunks)} chunk(s)")

    # ── 3 & 4. EMBED + STORE ───────────────────────────────
    # OpenAIEmbeddings calls the OpenAI API to turn each chunk into a vector.
    # Chroma.from_documents() does both steps in one call:
    #   • Embeds every chunk
    #   • Saves them to the local `chroma_db/` folder
    print("🧠  Embedding chunks and saving to ChromaDB …")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # Cheapest OpenAI embedding model
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="student_docs",
    )

    print(f"✅  Vector store ready at: {CHROMA_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an *existing* ChromaDB from disk (no re-embedding needed).
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # Cheapest OpenAI embedding model
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="student_docs",
    )

def get_retriever(vector_store: Chroma):
    """
    Convert the vector store into a LangChain Retriever.
    """

    return vector_store.as_retriever(
        search_type="similarity",   # Find the closest vectors
        search_kwargs={"k": 4},     # Return top 4 matching chunks
    )


if __name__ == "__main__":
    build_vector_store("data/Handbook.pdf")