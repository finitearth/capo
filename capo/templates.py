DOWNSTREAM_TEMPLATE = """<instruction>

<few_shots>


Input: """

FEWSHOT_TEMPLATE = """Input: <input>
Output: <output>"""

CROSSOVER_TEMPLATE = """You receive two prompts for the following task: <task_desc>
Please merge the two prompts into a single coherent prompt. Maintain the key linguistic features from both original prompts:
Prompt 1: <mother>
Prompt 2: <father>

Return the new prompt in the following format:
<prompt>new prompt</prompt>"""

MUTATION_TEMPLATE = """You receive a prompt for the following task: <task_desc>
Please rephrase the prompt, preserving its core meaning while substantially varying the linguistic style.
Prompt: <instruction>

Return the new prompt in the following format:
<prompt>new prompt</prompt>"""

EVOPROMPT_GA_SIMPLIFIED_TEMPLATE = """You receive two prompts for the following task: <task_desc>
1. Please merge the two prompts into a single coherent prompt. Maintain the key linguistic features from both original prompts:
Prompt 1: <prompt1>
Prompt 2: <prompt2>

2. Please rephrase the prompt generated in step 1, preserving its core meaning while substantially varying the linguistic style.
Return the final prompt in the following format:
<prompt>final prompt</prompt>"""
