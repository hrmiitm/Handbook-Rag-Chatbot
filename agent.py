from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful study assistant. Answer the student's question
using ONLY the context below. If the answer isn't in the context,
say "I couldn't find that in the uploaded document."

Keep your answer clear, friendly, and beginner-friendly.
Use bullet points when listing multiple things.

CONTEXT:
{context}
""",
    ),
    ("human", "{question}"),
])

def format_docs(docs: list) -> str:
    """
    Join retrieved Document chunks into a single string.

    Returns:
        str: A single formatted string with all chunk text.
    """
    s =  "\n\n".join(doc.page_content for doc in docs)
    print(s)
    return s


from chat_model import get_chat_model
from database import load_vector_store, get_retriever

llm = get_chat_model()
vs = load_vector_store()
retriever = get_retriever(vs)



question="Can i repeat foundation level courses in degree level?"

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} | RAG_PROMPT | llm
)

response = chain.invoke(question) # ---------------------
print("Question:", question)
print("-" * 50)
print("Answer:\n", response.content)