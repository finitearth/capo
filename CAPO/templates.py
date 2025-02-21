FEW_SHOT_TEMPLATE = """<instruction>

<examples>"""

DOWNSTREAM_TEMPLATE = """
<instruction>
Input: <input>
Output:
"""

CROSSOVER_TEMPLATE = """
Combine the following prompts:

Prompt 1: <mother>
Prompt 2: <father>

Return the result between <prompt> and </prompt>.
"""

MUTATION_TEMPLATE = """
Improve the prompt and return the result between <prompt> and </prompt>:

<instruction>
"""