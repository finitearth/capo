from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.llms.base_llm import BaseLLM
from promptolution.llms.api_llm import APILLM
from promptolution.tasks.base_task import BaseTask
from promptolution.tasks import ClassificationTask
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from collections import defaultdict
from typing import List, Tuple, Callable, Dict
import random
import numpy as np
import pandas as pd
import math


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


def hoeffdings_inequality_test_diff(
    score_a: float,
    score_b: float,
    n: int,
    delta: float = 0.05,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> bool:
    """
    Uses Hoeffding's inequality to test if candidate A's accuracy is significantly
    higher than candidate B's accuracy when they have different numbers of evaluations.

    For a candidate with n evaluations and observed average score, Hoeffding's inequality
    gives a confidence bound:
        epsilon = sqrt((R^2 * log(2/delta)) / (2*n))
    where R = max_val - min_val.

    Candidate A is considered significantly better than candidate B if:
        (score_a - epsilon_a) > (score_b + epsilon_b)

    Parameters:
        score_a (float): Observed average accuracy for candidate A (default range [0,1]).
        score_b (float): Observed average accuracy for candidate B.
        n (int): Number of independent evaluations.
        delta (float): Significance level (default 0.05 for 95% confidence).
        min_val (float): Minimum possible score (default 0.0).
        max_val (float): Maximum possible score (default 1.0).

    Returns:
        bool: True if candidate A is significantly better than candidate B, False otherwise.
    """
    R = max_val - min_val
    epsilon_a = math.sqrt((R**2 * math.log(2 / delta)) / (2 * n))
    epsilon_b = math.sqrt((R**2 * math.log(2 / delta)) / (2 * n))

    result = (score_a - epsilon_a) > (score_b + epsilon_b)

    return result


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
        examples_str = "\n".join(
            [
                f"Input: {sample_input}\nOutput: {response}"
                for sample_input, response in self.examples
            ]
        )
        prompt = FEW_SHOT_TEMPLATE.replace(
            "<instruction>", self.instruction_text
        ).replace("<examples>", examples_str)

        return prompt


class CAPOptimizer(BaseOptimizer):
    """
    Optimizer that evolves prompt instructions using crossover, mutation,
    and racing based on evaluation scores and statistical tests.
    """

    def __init__(
        self,
        initial_prompts: List[str],
        task: BaseTask,
        meta_llm: BaseLLM,
        downstream_llm: BaseLLM,
        length_penalty: float,
        block_size: int,
        crossovers_per_iter: int,
        upper_shots: int,
        max_n_blocks_eval: int,
        test_statistic: Callable,
        crossover_meta_prompt: str = None,
        mutation_meta_prompt: str = None,
        callbacks: List[Callable] = [],
        predictor: BasePredictor = None,
    ):
        """
        Initializes the CAPOptimizer with various parameters for prompt evolution.

        Parameters:
            initial_prompts (List[str]): Initial prompt instructions.
            task (BaseTask): The task instance containing dataset and description.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            downstream_llm (BaseLLM): The downstream language model used for responses.
            length_penalty (float): Penalty factor for prompt length.
            block_size (int): Number of samples per evaluation block.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            max_n_blocks_eval (int): Maximum number of evaluation blocks.
            test_statistic (Callable): Function to test significance between prompts.
                Inputs are (score_a, score_b, n_evals) and returns True if A is better.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): List of callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt performance.
        """
        # Pass initial_prompts and task to the base optimizer
        super().__init__(initial_prompts, task, callbacks, predictor)
        self.task = task

        self.meta_llm = meta_llm
        self.downstream_llm = downstream_llm

        self.crossover_meta_prompt = crossover_meta_prompt or CROSSOVER_TEMPLATE
        self.mutation_meta_prompt = mutation_meta_prompt or MUTATION_TEMPLATE

        self.population_size = len(initial_prompts)
        self.block_size = block_size
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = test_statistic

        self.blocks = self._split_into_blocks()
        self.population = self._initialize_population(initial_prompts)

        self.length_penalty = length_penalty

        # Caches evaluations: (prompt id, block id) -> score
        self.evaluation_cache: Dict[Tuple[int, int], float] = {}

    def _split_into_blocks(self) -> List[Tuple[int, np.ndarray]]:
        """
        Splits the task's dataset into blocks of indices for evaluation.

        Returns:
            List[Tuple[int, np.ndarray]]: List of tuples (block_id, indices array).
        """
        # Use the task's xs (and corresponding ys) to create index blocks.
        num_samples = len(self.task.xs)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        blocks = [
            indices[i : i + self.block_size]
            for i in range(0, num_samples, self.block_size)
        ]
        return list(enumerate(blocks))  # Each block is (block_id, indices)

    def _initialize_population(self, initial_prompts: List[str]) -> List[Prompt]:
        """
        Initializes the population of Prompt objects from initial instructions.

        Parameters:
            initial_prompts (List[str]): List of initial prompt instructions.

        Returns:
            List[Prompt]: Initialized population of prompts with few-shot examples.
        """
        population = []
        num_samples = len(self.task.xs)
        for instruction_text in initial_prompts:
            few_shots = self._create_few_shot_examples(instruction_text, num_samples)
            population.append(Prompt(instruction_text, few_shots))
        return population

    def _create_few_shot_examples(
        self, instruction: str, n_shots: int
    ) -> List[Tuple[str, str]]:
        num_examples = random.randint(0, self.upper_shots)
        all_indices = list(range(n_shots))
        selected_indices = random.sample(all_indices, num_examples)
        few_shots = []
        for idx in selected_indices:
            sample_input = self.task.xs[idx]
            # Assuming sample_input can be cast to string
            if random.random() < 0.5:
                response = self.downstream_llm.get_response(
                    [f"{instruction}\nInput: {sample_input}\nOutput:"]
                )[0]
            else:
                response = self.task.ys[idx]
            few_shots.append((str(sample_input), response))

        return few_shots

    def evaluate_prompt_on_block(
        self, prompt: Prompt, block_indices: np.ndarray, block_id: int
    ) -> float:
        """
        Evaluates a prompt on a given block of data and applies the length penalty.

        Parameters:
            prompt (Prompt): The prompt to evaluate.
            block_indices (np.ndarray): Array of indices for the current block.
            block_id (int): Identifier for the current block.

        Returns:
            float: The evaluation score (adjusted by prompt length penalty).
        """
        cache_key = (id(prompt), block_id)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        prompt_str = prompt.construct_prompt()
        prompt_length = len(prompt_str.split())
        # Extract the block's inputs and targets using the indices
        block_inputs = self.task.xs[block_indices]
        block_targets = self.task.ys[block_indices]

        # evaluate on task
        temp_task = ClassificationTask.from_dataframe(
            pd.DataFrame({"x": block_inputs, "y": block_targets}),
            description=self.task.description,
        )
        score = temp_task.evaluate([prompt.construct_prompt()], self.predictor)

        total_score = score - prompt_length * self.length_penalty
        self.evaluation_cache[cache_key] = total_score
        return total_score

    def _crossover(self, parents: List[Prompt]) -> List[Prompt]:
        """
        Performs crossover among parent prompts to generate offsprings.

        Parameters:
            parents (List[Prompt]): List of parent prompts.

        Returns:
            List[Prompt]: List of new offsprings after crossover.
        """
        offsprings = []
        for _ in range(self.crossovers_per_iter):
            mother, father = random.sample(parents, 2)
            crossover_prompt = (
                self.crossover_meta_prompt.replace("<mother>", mother.instruction_text)
                .replace("<father>}", father.instruction_text)
                .strip()
            )
            child_instruction = (
                self.meta_llm.get_response([crossover_prompt])[0]
                .split("<prompt>")[-1]
                .split("</prompt>")[0]
                .strip()
            )
            combined_examples = mother.examples + father.examples
            num_examples = int(len(mother.examples) * 0.5 + len(father.examples) * 0.5)
            child_examples = random.sample(
                combined_examples, min(num_examples, len(combined_examples))
            )

            offsprings.append(Prompt(child_instruction, child_examples))
        return offsprings

    def _mutate(self, offsprings: List[Prompt]) -> List[Prompt]:
        """
        Applies mutation to offsprings to generate new candidate prompts.

        Parameters:
            offsprings (List[Prompt]): List of offsprings to mutate.

        Returns:
            List[Prompt]: List of mutated prompts.
        """
        mutated = []
        for prompt in offsprings:
            mutation_prompt = self.mutation_meta_prompt.replace(
                "<instruction>", prompt.instruction_text
            )
            new_instruction = (
                self.meta_llm.get_response([mutation_prompt])[0]
                .split("<prompt>")[-1]
                .split("</prompt>")[0]
                .strip()
            )
            num_fewshots = random.randint(0, self.upper_shots)
            new_few_shots = self._create_few_shot_examples(
                new_instruction, num_fewshots
            )
            # Optionally combine some existing examples from the prompt
            num_old = max(0, num_fewshots - len(new_few_shots))
            old_examples = random.sample(
                prompt.examples, min(num_old, len(prompt.examples))
            )

            combined_examples = old_examples + new_few_shots
            random.shuffle(combined_examples)
            mutated.append(Prompt(new_instruction, combined_examples))
        return mutated

    def _do_racing(self, candidates: List[Prompt], k: int) -> List[Prompt]:
        """
        Performs the racing (selection) phase by comparing candidates based on their
        evaluation scores using the provided test statistic.

        Parameters:
            candidates (List[Prompt]): List of candidate prompts.
            k (int): Number of survivors to retain.

        Returns:
            List[Prompt]: List of surviving prompts after racing.
        """
        prompt_evaluations = defaultdict(lambda: [])
        for block_id, block_indices in self.blocks:
            for prompt in candidates:
                score = self.evaluate_prompt_on_block(prompt, block_indices, block_id)
                pid = id(prompt)
                prompt_evaluations[pid].append(score)

            candidate_avg_scores = {
                id(prompt): np.mean(prompt_evaluations[id(prompt)])
                for prompt in candidates
            }
            candidate_n_evals = {
                id(prompt): len(prompt_evaluations[id(prompt)]) * self.block_size
                for prompt in candidates
            }
            survivors = []
            for candidate in candidates:
                candidate_id = id(candidate)
                score = candidate_avg_scores[candidate_id]
                n_better = sum(
                    1
                    for other in candidates
                    if self.test_statistic(
                        score,
                        candidate_avg_scores[id(other)],
                        candidate_n_evals[candidate_id],
                    )
                )
                if n_better < k:
                    survivors.append(candidate)
            candidates = survivors
            if len(candidates) <= k or block_id == self.max_n_blocks_eval:
                break
        return candidates[:k]

    def optimize(self, n_steps: int) -> List[Prompt]:
        """
        Main optimization loop that evolves the prompt population over a number of steps.

        Parameters:
            n_steps (int): Number of optimization steps to perform.

        Returns:
            List[Prompt]: The final population of prompts after optimization.
        """
        for _ in range(n_steps):
            offsprings = self._crossover(self.population)
            mutated = self._mutate(offsprings)
            combined = self.population + mutated
            self.population = self._do_racing(combined, self.population_size)
            self._on_step_end()
        self._on_train_end()
        return self.population


if __name__ == "__main__":
    token = open("deepinfratoken.txt", "r").read()

    meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
    downstream_llm = meta_llm
    # Create the task – note that the task already loads its dataset.
    df = pd.read_csv(
        "hf://datasets/nakamoto-yama/dt-mappings/yama_dt_mappings.csv"
    ).rename({"Degree Type": "x", "Mapping": "y"}, axis=1)
    task = ClassificationTask.from_dataframe(df, description="test test")
    # Ensure task.xs and task.ys are set (here, converting DataFrame columns to numpy arrays)
    task.xs = df["x"].to_numpy()
    task.ys = df["y"].to_numpy()

    initial_prompts = [
        create_prompts_from_samples(task, downstream_llm) for _ in range(10)
    ]

    predictor = Classificator(downstream_llm, df["y"].unique())
    test_statistic = lambda x, y, n: hoeffdings_inequality_test_diff(x, y, n, delta=0.5)

    optimizer = CAPOptimizer(
        initial_prompts=initial_prompts,
        task=task,
        meta_llm=meta_llm,
        downstream_llm=downstream_llm,
        length_penalty=0.01,
        block_size=30,
        crossovers_per_iter=2,
        upper_shots=5,
        max_n_blocks_eval=5,
        test_statistic=test_statistic,
        predictor=predictor,
    )
    best_prompts = optimizer.optimize(n_steps=3)
    print(f"Best instructions:\n\n{[p.construct_prompt() for p in best_prompts]}")
