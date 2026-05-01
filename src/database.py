import os
import hashlib

from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DATA_DIR = "data"
DATA_URL_FILE_NAME = 'resources.txt'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "student_docs"
CHROMA_DIR = "chroma_db"

def load_all_documents(data_dir):
    all_doc = []

    local_file_paths = []
    web_files_urls = []

    # Find local and web files path
    for file_name in os.listdir(data_dir): # list of files = [Handbook, resource]
        file_path = os.path.join(data_dir, file_name)

        if file_name != DATA_URL_FILE_NAME:
            local_file_paths.append(file_path)
        else:
            with open(file_path) as f:
                urls = [line.strip() for line in f if line.strip()] # ["www.g", "q,.."]
                web_files_urls.extend(urls)

    # Load local and web files
    print(local_file_paths, web_files_urls)
    
    for url in web_files_urls:
        url_loader = UnstructuredLoader(
            web_url=url, 
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap = CHUNK_OVERLAP
            )
        all_doc.extend(url_loader.load())

    # Load local files
    if local_file_paths:
        local_loader = UnstructuredLoader(
            file_path=local_file_paths, 
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap = CHUNK_OVERLAP
            ) 
        all_doc.extend(local_loader.load())

    return all_doc


def get_embedding_function():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_DIR,  
    )

# Store in Vector Db unique chunk
def _doc_id(doc):
    """Generate unique ID from content + source for deduplication."""
    # Better to hash the full content to avoid collisions on similar starting text
    s = f"{doc.metadata.get('source','')}:{doc.page_content}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def add_documents_to_store(vs, chunks):
    """Add chunks with deduplication — safe to run multiple times."""
    existing = set(vs.get()["ids"] or [])
    
    unique_new_chunks = []
    unique_new_ids = []
    seen_in_batch = set()
    
    for chunk in chunks:
        c_id = _doc_id(chunk)
        
        # Keep chunk only if it's not in the DB AND not already in this batch
        if c_id not in existing and c_id not in seen_in_batch:
            unique_new_chunks.append(chunk)
            unique_new_ids.append(c_id)
            seen_in_batch.add(c_id)
            
    if unique_new_chunks:
        vs.add_documents(unique_new_chunks, ids=unique_new_ids)
        print(f"✅ Added {len(unique_new_chunks)} (skipped {len(chunks)-len(unique_new_chunks)})")
    else:
        print("ℹ️  All docs already exist")

def get_store_stats(vs):
    """Get stats about the vector store."""
    data = vs.get()
    sources = {m.get("source","?") for m in data["metadatas"]}
    return {"total": len(data["ids"]), "sources": sorted(sources)}

if __name__ == "__main__":
    # print("---Doc Loading---")
    # docs = load_all_documents(DATA_DIR)
    # print("Number of Doc Loaded", len(docs), type(docs), type(docs[0]))
    # print(docs)

    # print(get_vector_store())

    vs = get_vector_store()

    chunks = load_all_documents(DATA_DIR)

    add_documents_to_store(vs, chunks)

    stats = get_store_stats(vs)

    print(f"📈 {stats['total']} chunks from {len(stats['sources'])} sources")
