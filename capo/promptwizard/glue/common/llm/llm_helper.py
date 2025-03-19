from llama_index.core.llms import LLM
from llama_index.core.callbacks.token_counting import TokenCountingHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler


def get_token_counter(llm_handle: LLM) -> TokenCountingHandler:
    """
    Extract TokenCountingHandler handler from llm_handle.

    :param llm_handle: Object of class LLM, which is the handle to make all LLM related calls
    :return: Object of TokenCountingHandler, that's registered as callback_manager in LLM. If not found, return None
    """
    return get_callback_handler(llm_handle, "TokenCountingHandler")


def get_callback_handler(llm_handle: LLM, class_name: str) -> BaseCallbackHandler:
    """
    Extract callback_manager from llm_handle, find out which call back manager is of class type `class_name`.
    Return that object.

    :param llm_handle: Object of class LLM, which is the handle to make all LLM related calls
    :param class_name: Name of class (without prefix file path) e.g. TokenCountingHandler
    :return: Object of BaseCallbackHandler, that's registered as callback_manager in LLM. If not found, return None
    """
    if llm_handle and llm_handle.callback_manager:
        for handler in llm_handle.callback_manager.handlers:
            if type(handler).__name__ == class_name:
                return handler

    return None
