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

MUTATION_TEMPLATE = """Merge the following two sentences into a single coherent sentence. Maintain the key linguistic features from both original sentences:
Prompt 1: <prompt1>
Prompt 2: <prompt2>

Return the new instruction in the following format:
<prompt>new instruction</prompt>"""
