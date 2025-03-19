from dataclasses import dataclass
from typing import List

from ....common.base_classes import UniversalBaseClass
from ...constants import PromptOptimizationParams, PromptPool


@dataclass
class CritiqueNRefinePromptPool(PromptPool):
    quest_reason_ans: str
    expert_profile: str
    ans_delimiter_instruction: str
    intent_template: str
    thinking_styles: List[str]
    meta_critique_template: str
    meta_positive_critique_template: str
    critique_refine_template: str
    solve_template: str
    examples_critique_template: str
    examples_optimization_template: str
    meta_sample_template: str
    intent_template: str
    expert_template: str
    generate_reason_template: str
    reason_optimization_template: str
    examples_critique_template_zero_shot: str


@dataclass
class CritiqueNRefineParams(PromptOptimizationParams, UniversalBaseClass):
    unique_model_id: str
    # Number of candidate prompts to generate in given iteration
    style_variation: int
    # Number of questions to be asked to LLM in a single go
    questions_batch_size: int
    # Number of batches of questions to correctly answered, for a prompt to be considered as performing good
    min_correct_count: int
    # Max number of mini-batches on which we should evaluate our prompt
    max_eval_batches: int
    # Number of top best performing prompts to be considered for next iterations
    top_n: int
    # Number of rounds of mutation to be performed when generating different styles
    mutation_rounds: int
    # Refine instruction post mutation
    refine_instruction: bool
    # Number of iterations for conducting <mutation_rounds>  rounds of mutation of task description
    # followed by refinement of instructions
    mutate_refine_iterations: int
    # Number of iterations for refining task description and in context examples for few-shot
    refine_task_eg_iterations: int
    # Description of task. This will be fed to prompt
    task_description: str
    # Base instruction, in line with your dataset. This will be fed to prompt
    base_instruction: str
    # Instruction for specifying answer format
    answer_format: str
    # Number of samples from dataset, set aside as training data. In every iteration we would be drawing
    # `questions_batch_size` examples from training data with replacement.
    seen_set_size: int
    # Number of examples to be given for few shots
    few_shot_count: int
    # Generate synthetic reasoning
    generate_reasoning: bool
    # Generate description of an expert which can solve the task at hand
    generate_expert_identity: bool
    # Generate keywords that describe the intent of the task
    generate_intent_keywords: bool
    # number of synthetic training examples to be generated
    num_train_examples: int
