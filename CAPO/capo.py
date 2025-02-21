from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.llms.base_llm import BaseLLM
from promptolution.tasks.base_task import BaseTask
from promptolution.tasks import ClassificationTask
from promptolution.predictors.base_predictor import BasePredictor

from capo.templates import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE
from capo.utils import Prompt

from collections import defaultdict
from typing import List, Tuple, Callable, Dict
import random
import numpy as np
import pandas as pd



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
        few_shot_split_size: float,
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
            few_shot_split_size (float): Fraction of dataset to use for few-shots.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt
                performance.
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

        # Each block is (block_id, indices), few_shot_split is a list of indices
        self.blocks, self.few_shot_indices = self._split_dataset(few_shot_split_size)
        self.population = self._initialize_population(initial_prompts)

        self.length_penalty = length_penalty

        # Caches evaluations: (prompt id, block id) -> score
        self.evaluation_cache: Dict[Tuple[int, int], float] = {}

    def _split_dataset(
        self, few_shot_split_size: float
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Splits the task's dataset into blocks of indices for evaluation
        and few-shot examples.

        Returns:
            List[Tuple[int, np.ndarray]]: List of tuples (block_id, indices array),
            Tuple[int, np.ndarray]: List of indices for few-shot examples.
        """
        # Use the task's xs (and corresponding ys) to create index blocks.
        num_samples = len(self.task.xs)
        indices = list(range(num_samples))
        random.shuffle(indices)

        n_few_shots = int(few_shot_split_size * num_samples)
        few_shot_indices = indices[:n_few_shots]
        blocks = [
            indices[i : i + self.block_size]
            for i in range(n_few_shots, num_samples, self.block_size)
        ]
        return list(enumerate(blocks)), few_shot_indices

    def _initialize_population(self, initial_prompts: List[str]) -> List[Prompt]:
        """
        Initializes the population of Prompt objects from initial instructions.

        Parameters:
            initial_prompts (List[str]): List of initial prompt instructions.

        Returns:
            List[Prompt]: Initialized population of prompts with few-shot examples.
        """
        population = []
        for instruction_text in initial_prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = self._create_few_shot_examples(instruction_text, num_examples)
            population.append(Prompt(instruction_text, few_shots))
        return population

    def _create_few_shot_examples(
        self, instruction: str, num_examples: int
    ) -> List[Tuple[str, str]]:
        selected_indices = random.sample(self.few_shot_indices, num_examples)
        few_shots = []
        for idx in selected_indices:
            sample_input = self.task.xs[idx]
            # in half of the cases generate reasoning from downstream model
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
            num_examples = int((len(mother.examples) + len(father.examples)) / 2)
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
            # combine the new shots with some existing from the prompt
            old_examples = random.sample(
                prompt.examples,
                min(num_fewshots - len(new_few_shots), len(prompt.examples)),
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

            candidate_scores = {
                id(prompt): prompt_evaluations[id(prompt)] for prompt in candidates
            }

            survivors = []
            for candidate in candidates:
                candidate_id = id(candidate)
                scores = candidate_scores[candidate_id]
                n_better = sum(
                    1
                    for other in candidates
                    if self.test_statistic(
                        scores,
                        candidate_scores[id(other)],
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
        Main optimization loop that evolves the prompt population.

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

        prompts = [p.construct_prompt() for p in self.population]
        return prompts
 
