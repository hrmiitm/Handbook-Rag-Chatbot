from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from chat_model import get_chat_model
from database import get_retriever


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful study assistant. Answer the student's question
using ONLY the context below. If the answer isn't in the context,
say "I couldn't find that in the uploaded document."

Keep your answer clear, friendly, and beginner-friendly.
Use bullet points when listing multiple things. And also quote the sources/context info that is relevant at end of final response

Make sure your response must be based on facts only

---
CONTEXT:
{context}
""",
    ),
    ("human", "{question}"),
])


def format_docs_with_sources(docs):
    """Format docs with source info so LLM can cite them."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        url = doc.metadata.get("url", "N/A")
        formatted.append(f"[Source {i+1}: {source}, Page {page}, URL {url}]\n{doc.page_content}")
    s = "\n\n---\n\n".join(formatted)
    print(s)
    print(docs)
    print("---"*20)
    return s



llm = get_chat_model()
retriever = get_retriever()

question = "Tell me about the credit requirements for 4 year?" 

chain = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()} | RAG_PROMPT | llm
)

response = chain.invoke(question)
print("Answer:\n", response.content)