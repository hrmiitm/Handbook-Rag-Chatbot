from langchain_community.document_loaders import UnstructuredURLLoader

# Define the URL
url = "https://docs.google.com/document/d/e/2PACX-1vSUvKzH7yIXNVwUgRYSIT8M0x1jhFSkslEtj9UPo3dtWI_sJ38Hh_PzbBygpF0vIOo8K7lTy-uYkqdu/pub"

# Use UnstructuredURLLoader for web-hosted content
loader = UnstructuredURLLoader(urls=[url])

docs = loader.load()

# Print the content of the first document
print(docs[0].page_content)