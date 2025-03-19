from .utils.logging import get_glue_logger

logger = get_glue_logger(__name__)


class GlueException(Exception):
    """
    Base class for all exceptions in Glue framework
    """
    def __init__(self, err_message):
        logger.error(f"\n Error: {err_message}\n")
        super().__init__(err_message)


class GlueLLMException(GlueException):
    """
    Base class for all exceptions related to LLM
    """
    def __init__(self, err_message, excep_obj):
        message = ("LLM exception\n"
                   f"Exception: {err_message}\n"
                   f"Exception logs: {excep_obj}")

        super().__init__(message)


class GlueValidaionException(GlueException):
    """
    Base class for all exceptions related to Validation in Glue framework
    """
    def __init__(self, err_message, excep_obj):
        message = ("[Invalid user input detected]\n"
                   f"Exception: {err_message}\n"
                   f"Exception logs: {excep_obj}")

        super().__init__(message)

