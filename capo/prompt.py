"""
Defines the Prompt class for structuring prompts with instructions and few-shot examples.
Provides functionality to construct complete prompts using templates and handle proper formatting.
"""

from typing import List

from capo.templates import DOWNSTREAM_TEMPLATE


class Prompt:
    """
    Represents a prompt consisting of an instruction and few-shot examples.
    """

    def __init__(self, instruction_text: str, few_shots: List[str]):
        """
        Initializes the Prompt with an instruction and associated examples.

        Parameters:
            instruction_text (str): The instruction or prompt text.
            few_shots (List[str]): List of examples as string.
        """
        self.instruction_text = instruction_text.strip()
        self.few_shots = few_shots

    def construct_prompt(self) -> str:
        """
        Constructs the full prompt string by replacing placeholders in the template
        with the instruction and formatted examples.

        Returns:
            str: The constructed prompt string.
        """
        few_shot_str = "\n\n".join(self.few_shots).strip()
        prompt = (
            DOWNSTREAM_TEMPLATE.replace("<instruction>", self.instruction_text)
            .replace("<few_shots>", few_shot_str)
            .replace("\n\n\n\n", "\n\n")  # replace extra newlines if no few shots are provided
            .strip()
        )
        return prompt

    def __str__(self):
        return self.construct_prompt()
