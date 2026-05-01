from langchain.chat_models import init_chat_model


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURE = 0.1 # To Be Grounded to facts
DEFAULT_MAX_TOKENS = 2000


def get_chat_model(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):

    return init_chat_model(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
    )