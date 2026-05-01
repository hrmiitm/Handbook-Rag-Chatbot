from langchain_unstructured import UnstructuredLoader
import os

DATA_DIR = "data"
DATA_URL_FILE_NAME = 'resources.txt'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

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


def get_vector_store():
    pass

if __name__ == "__main__":
    print("---Doc Loading---")
    docs = load_all_documents(DATA_DIR)
    print("Number of Doc Loaded", len(docs), type(docs), type(docs[0]))
    # print(docs)

