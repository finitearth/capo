import hashlib
import random
import string
from typing import List

from capo.templates import FEW_SHOT_TEMPLATE


def generate_hash_from_string(text: str):
    """Generate a hash from a string."""
    hash_object = hashlib.sha256(text.encode())
    hash_string = hash_object.hexdigest()
    return hash_string


def generate_random_hash():
    """Generate a random hash."""
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=32))

    return generate_hash_from_string(random_string)


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
        self.few_shots = few_shots  # List of (sample_input, response)

    def construct_prompt(self) -> str:
        """
        Constructs the full prompt string by replacing placeholders in the template
        with the instruction and formatted examples.

        Returns:
            str: The constructed prompt string.
        """
        few_shot_str = "\n\n".join(self.few_shots).strip()
        prompt = FEW_SHOT_TEMPLATE.replace("<instruction>", self.instruction_text.strip()).replace(
            "<examples>", few_shot_str
        )
        return prompt

    def __str__(self):
        return self.construct_prompt()
