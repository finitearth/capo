class GlueLLM:
    """
    Abstract class that can be inherited by a class that defines Custom LLM
    """

    @staticmethod
    def get_tokenizer():
        """
        This method should either return an encode method of tokenizer or None
        :return: method

        e.g. When using HuggingFace tokenizer
        tokenizer = Tokenizer(BPE())
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        return fast_tokenizer.encode

        e.g. When using tiktoken tokenizer
        return tiktoken.encoding_for_model(azure_oai_model.model_name_in_azure).encode
        """
        return None
