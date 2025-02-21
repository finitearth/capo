import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.llms.base_llm import BaseLLM
from promptolution.tasks.base_task import BaseTask
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks import ClassificationTask
from promptolution.tasks.base_task import BaseTask

from capo.templates import CROSSOVER_TEMPLATE, MUTATION_TEMPLATE, DOWNSTREAM_TEMPLATE
from capo.utils import Prompt
from capo.task import CAPOTask

from typing import List, Tuple, Callable, Dict
import random
import numpy as np
from itertools import compress


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
        shuffle_blocks_per_iter: bool = True,
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
            shuffle_blocks_per_iter (bool, optional): Whether to shuffle blocks each
                iteration. Defaults to True.
            few_shot_split_size (float): Fraction of dataset to use for few-shots.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt
                performance.
        """
        if not isinstance(task, CAPOTask):
            task = CAPOTask.from_task(task, few_shot_split_size, block_size)

        # Pass initial_prompts and task to the base optimizer
        super().__init__(initial_prompts, task, callbacks, predictor)

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

        self.shuffle_blocks_per_iter = shuffle_blocks_per_iter
        self.prompts = self._initialize_population(initial_prompts)

        self.length_penalty = length_penalty

        # Caches evaluations: (prompt id, block id) -> score
        self.evaluation_cache: Dict[Tuple[int, int], float] = {}

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
        selected_indices = random.sample(self.task.few_shots, num_examples)
        few_shots = []
        for idx in selected_indices:
            sample_input = self.task.xs[idx]
            sample_target = self.task.ys[idx]
            # in half of the cases generate reasoning from downstream model
            if random.random() < 0.5:
                response = self.downstream_llm.get_response(
                    [
                        DOWNSTREAM_TEMPLATE
                        .replace("<input>", sample_input)
                        .replace("<instruction>", instruction)
                    ],
                )[0]
            else:
                response = sample_target
            few_shots.append((str(sample_input), response))

        return few_shots

    def _crossover(self, parents: List[Prompt]) -> List[Prompt]:
        """
        Performs crossover among parent prompts to generate offsprings.

        Parameters:
            parents (List[Prompt]): List of parent prompts.

        Returns:
            List[Prompt]: List of new offsprings after crossover.
        """
        crossover_prompts = []
        offspring_few_shots = []
        for _ in range(self.crossovers_per_iter):
            mother, father = random.sample(parents, 2)
            crossover_prompt = (
                self.crossover_meta_prompt.replace("<mother>", mother.instruction_text)
                .replace("<father>", father.instruction_text)
                .strip()
            )
            crossover_prompts.append(crossover_prompt)
            combined_few_shots = mother.examples + father.examples
            num_few_shots = int((len(mother.examples) + len(father.examples)) / 2)
            offspring_few_shot = random.sample(
                combined_few_shots, min(num_few_shots, len(combined_few_shots))
            )
            offspring_few_shots.append(offspring_few_shot)

        child_instructions = self.meta_llm.get_response(crossover_prompts)

        offsprings = []
        for instruction, examples in zip(child_instructions, offspring_few_shots):
            instruction = (
                instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            )
            offsprings.append(Prompt(instruction, examples))
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
            new_few_shots = self._create_few_shot_examples(new_instruction, num_fewshots)
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
        if self.shuffle_blocks_per_iter:
            random.shuffle(self.task.blocks)

        for block_id, _ in self.task.blocks:
            scores = self.task.evaluate_on_block(
                [c.construct_prompt() for c in candidates], block_id, self.predictor
            )

            comparison_matrix = np.array(
                [
                    [self.test_statistic(score, other_score) for other_score in scores]
                    for score in scores
                ]
            )
            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)
            # Create mask for survivors and filter candidates
            candidates = list(compress(candidates, n_better < k))
            if len(candidates) <= k or block_id == self.max_n_blocks_eval:
                break

        self.scores = list(range(k))
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
            offsprings = self._crossover(self.prompts)
            mutated = self._mutate(offsprings)
            combined = self.prompts + mutated
            self.prompts = self._do_racing(combined, self.population_size)
            self._on_step_end()
        self._on_train_end()

        prompts = [p.construct_prompt() for p in self.prompts]
        return prompts
