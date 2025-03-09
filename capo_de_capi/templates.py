FEW_SHOT_TEMPLATE = """<instruction>

<examples>"""

CROSSOVER_TEMPLATE = """Merge the following two sentences into a single coherent sentence. Maintain the key linguistic features from both original sentences:
Prompt 1: <mother>
Prompt 2: <father>

Return the new instruction in the following format:
<prompt>new instruction</prompt>"""

MUTATION_TEMPLATE = """Please do the following for this task: <task_desc>
Rephrase the following instruction while preserving its core meaning, while substantially differing in linguistic style.
Return the new prompt between <prompt> and </prompt> tags.

<instruction>"""
