from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../../data/Handbook.pdf")
docs = loader.load()

print(len(docs))
print(docs[:3])