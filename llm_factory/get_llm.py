from llama_index.llms.ollama import Ollama

from config.settings import Settings

settings = Settings()
OLLAMA_URL = settings.OLLAMA_URL


# Module level cache for model and instance
_current_model_name = None
_current_llm_instance = None


def get_llm(model_name: str):
    global _current_model_name, _current_llm_instance

    # Check if the requested model is the same as the cached one
    if model_name == _current_model_name and _current_llm_instance is not None:
        return _current_llm_instance

    # If not, create a new instance and update the cache
    llm_instance = Ollama(base_url=OLLAMA_URL, model=model_name)
    _current_model_name = model_name
    _current_llm_instance = llm_instance

    return llm_instance


# Example:
# check_llm = get_llm(model_name="llama3:latest")
# print(check_llm)
# print(type(check_llm))