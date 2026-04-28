from langchain_openai import ChatOpenAI

def get_chat_model():
    llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0.1, # We want precise, factual answers
            )
    return llm