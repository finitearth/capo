from typing import List, Tuple

from capo.templates import FEW_SHOT_TEMPLATE


class Prompt:
    """
    Represents a prompt consisting of an instruction and few-shot examples.
    """

    def __init__(self, instruction_text: str, examples: List[Tuple[str, str]]):
        """
        Initializes the Prompt with an instruction and associated examples.

        Parameters:
            instruction_text (str): The instruction or prompt text.
            examples (List[Tuple[str, str]]): List of examples as (input, response).
        """
        self.instruction_text = instruction_text
        self.examples = examples  # List of (sample_input, response)

    def construct_prompt(self) -> str:
        """
        Constructs the full prompt string by replacing placeholders in the template
        with the instruction and formatted examples.

        Returns:
            str: The constructed prompt string.
        """
        examples_str = "\n\n".join(
            [
                f"Input: {sample_input}\nOutput: {response}"
                for sample_input, response in self.examples
            ]
        )
        prompt = FEW_SHOT_TEMPLATE.replace("<instruction>", self.instruction_text).replace(
            "<examples>", examples_str
        )

        return prompt
    
    def __str__(self):
        return self.construct_prompt()