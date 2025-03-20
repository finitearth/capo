import random
from itertools import compress
from logging import getLogger
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from promptolution.llms.base_llm import BaseLLM
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.tasks.base_task import BaseTask

from capo.prompt import Prompt
from capo.task import CAPOClassificationTask
from capo.templates import CROSSOVER_TEMPLATE, FEWSHOT_TEMPLATE, MUTATION_TEMPLATE


class CAPOptimizer(BaseOptimizer):
    """
    Optimizer that evolves prompt instructions using crossover, mutation,
    and racing based on evaluation scores and statistical tests.
    """

    def __init__(
        self,
        initial_prompts: List[str],
        task: BaseTask,
        df_few_shots: pd.DataFrame,
        meta_llm: BaseLLM,
        downstream_llm: BaseLLM,
        length_penalty: float,
        crossovers_per_iter: int,
        upper_shots: int,
        max_n_blocks_eval: int,
        test_statistic: Callable,
        shuffle_blocks_per_iter: bool = True,
        crossover_meta_prompt: str = None,
        mutation_meta_prompt: str = None,
        callbacks: List[Callable] = [],
        predictor: BasePredictor = None,
        verbosity: int = 0,
        logger=getLogger(__name__),
    ):
        """
        Initializes the CAPOptimizer with various parameters for prompt evolution.

        Parameters:
            initial_prompts (List[str]): Initial prompt instructions.
            task (BaseTask): The task instance containing dataset and description.
            df_few_shots (pd.DataFrame): DataFrame containing few-shot examples.
            meta_llm (BaseLLM): The meta language model for crossover/mutation.
            downstream_llm (BaseLLM): The downstream language model used for responses.
            length_penalty (float): Penalty factor for prompt length.
            crossovers_per_iter (int): Number of crossover operations per iteration.
            upper_shots (int): Maximum number of few-shot examples per prompt.
            p_few_shot_reasoning (float): Probability of generating llm-reasoning for few-shot examples, instead of simply using input-output pairs.
            n_trials_generation_reasoning (int): Number of trials to generate reasoning for few-shot examples.
            max_n_blocks_eval (int): Maximum number of evaluation blocks.
            test_statistic (Callable): Function to test significance between prompts.
                Inputs are (score_a, score_b, n_evals) and returns True if A is better.
            shuffle_blocks_per_iter (bool, optional): Whether to shuffle blocks each
                iteration. Defaults to True.
            crossover_meta_prompt (str, optional): Template for crossover instructions.
            mutation_meta_prompt (str, optional): Template for mutation instructions.
            callbacks (List[Callable], optional): Callbacks for optimizer events.
            predictor (BasePredictor, optional): Predictor to evaluate prompt
                performance.
            verbosity (int, optional): Verbosity level for logging. Defaults to 0.
        """
        assert isinstance(task, CAPOClassificationTask), "CAPOptimizer requires a CAPO task."

        super().__init__(initial_prompts, task, callbacks, predictor)
        self.df_few_shots = df_few_shots
        self.meta_llm = meta_llm
        self.downstream_llm = downstream_llm

        if hasattr(self.downstream_llm, "tokenizer"):
            self.token_count = lambda x: len(
                self.downstream_llm.tokenizer(x.construct_prompt())["input_ids"]
            )
        else:
            logger.warning(
                "The downstream language model does not have a tokenizer. Using simple token count."
            )
            self.token_count = lambda x: len(x.construct_prompt().split())

        self.crossover_meta_prompt = crossover_meta_prompt or CROSSOVER_TEMPLATE
        self.mutation_meta_prompt = mutation_meta_prompt or MUTATION_TEMPLATE

        self.population_size = len(initial_prompts)
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = test_statistic

        self.shuffle_blocks_per_iter = shuffle_blocks_per_iter

        self.length_penalty = length_penalty
        self.verbosity = verbosity
        self.logger = logger

        self.prompt_objects = self._initialize_population(initial_prompts)
        self.prompts = [p.construct_prompt() for p in self.prompt_objects]
        self.max_prompt_length = max(self.token_count(p) for p in self.prompt_objects)

        self.scores = np.empty(0)

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

        if self.verbosity > 0:
            self.logger.warning(
                f"🍿Initialized population with {len(population)} prompts: \n {[p.construct_prompt() for p in population]}"
            )
        return population

    def _create_few_shot_examples(
        self, instruction: str, num_examples: int
    ) -> List[Tuple[str, str]]:
        if num_examples == 0:
            return []
        few_shot_samples = self.df_few_shots.sample(num_examples, replace=False)
        sample_inputs = few_shot_samples["input"].values
        sample_targets = few_shot_samples["target"].values
        few_shots = [
            FEWSHOT_TEMPLATE.replace("<input>", i).replace(
                "<output>", f"{self.predictor.begin_marker}{t}{self.predictor.end_marker}"
            )
            for i, t in zip(sample_inputs, sample_targets)
        ]
        # Select partition of the examples to generate reasoning from downstream model
        preds, seqs = self.predictor.predict(
            instruction,
            sample_inputs,
            return_seq=True,
        )
        preds, seqs = preds.reshape(num_examples), seqs.reshape(num_examples)

        # Check which predictions are correct and get a single one per example
        for j in range(num_examples):
            # Process and clean up the generated sequences
            seqs[j] = seqs[j].replace(sample_inputs[j], "").strip()
            # Check if the prediction is correct and add reasoning if so
            if preds[j] == sample_targets[j]:
                few_shots[j] = FEWSHOT_TEMPLATE.replace("<input>", sample_inputs[j]).replace(
                    "<output>", seqs[j]
                )

        if self.verbosity > 1:
            self.logger.warning(f"🔫Few-shot examples: {few_shots}")
            self.logger.warning(f"💆‍♂️Generated reasoning: {seqs}")

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
                .replace("<task_desc>", self.task.description)
                .strip()
            )
            # collect all crossover prompts then pass them bundled to the meta llm (speedup)
            crossover_prompts.append(crossover_prompt)
            combined_few_shots = mother.few_shots + father.few_shots
            num_few_shots = (len(mother.few_shots) + len(father.few_shots)) // 2
            offspring_few_shot = random.sample(combined_few_shots, num_few_shots)
            offspring_few_shots.append(offspring_few_shot)

        child_instructions = self.meta_llm.get_response(crossover_prompts)
        if self.verbosity > 1:
            self.logger.warning(f"🥐Generated crossover prompts: \n{child_instructions}")

        offsprings = []
        for instruction, examples in zip(child_instructions, offspring_few_shots):
            instruction = instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
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
        # collect all mutation prompts then pass them bundled to the meta llm (speedup)
        mutation_prompts = [
            self.mutation_meta_prompt.replace("<instruction>", prompt.instruction_text).replace(
                "<task_desc>", self.task.description
            )
            for prompt in offsprings
        ]
        new_instructions = self.meta_llm.get_response(mutation_prompts)

        mutated = []
        for new_instruction, prompt in zip(new_instructions, offsprings):
            new_instruction = new_instruction.split("<prompt>")[-1].split("</prompt>")[0].strip()
            p = random.random()

            if p < 1 / 3 and len(prompt.few_shots) < self.upper_shots:  # add a random few shot
                new_few_shot = self._create_few_shot_examples(new_instruction, 1)
                new_few_shots = prompt.few_shots + new_few_shot
            if 1 / 3 <= p < 2 / 3 and len(prompt.few_shots) > 0:  # remove a random few shot
                new_few_shots = random.sample(prompt.few_shots, len(prompt.few_shots) - 1)
            else:  # do not change few shots, but shuffle
                new_few_shots = prompt.few_shots

            random.shuffle(new_few_shots)
            mutated.append(Prompt(new_instruction, new_few_shots))

        if self.verbosity > 0:
            self.logger.warning(f"🧟Generated {len(mutated)} mutated prompts.")
            self.logger.warning(f"😶Generated Prompts: {[p.construct_prompt() for p in mutated]}")

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

        block_scores = []
        for i, (block_id, _) in enumerate(self.task.blocks):
            # new_scores shape: (n_candidates, n_samples)
            new_scores = self.task.evaluate_on_block(
                [c.construct_prompt() for c in candidates], block_id, self.predictor
            )

            # subtract length penalty
            prompt_lengths = np.array([self.token_count(c) for c in candidates])
            rel_prompt_lengths = prompt_lengths / self.max_prompt_length

            new_scores = new_scores - self.length_penalty * rel_prompt_lengths[:, None]
            block_scores.append(new_scores)
            scores = np.concatenate(block_scores, axis=1)

            # boolean matrix C_ij indicating if candidate j is better than candidate i
            comparison_matrix = np.array(
                [
                    [self.test_statistic(other_score, score) for other_score in scores]
                    for score in scores
                ]
            )

            # Sum along rows to get number of better scores for each candidate
            n_better = np.sum(comparison_matrix, axis=1)

            if self.verbosity > 1:
                self.logger.warning(f"🛝Comparison Matrix: \n{comparison_matrix}")
                self.logger.warning(f"🔢Number of better scores: {n_better}")

                eliminated_prompts = [
                    c.construct_prompt() for c in compress(candidates, n_better >= k)
                ]
                eliminated_scores = scores[n_better >= k]
                self.logger.warning("⚰️Eliminated Prompts:")
                self.logger.warning(
                    "\n\n".join(
                        [
                            f"Prompt: {p} \n Score: {s}"
                            for p, s in zip(eliminated_prompts, eliminated_scores)
                        ]
                    )
                )

            # Create mask for survivors and filter candidates
            candidates = list(compress(candidates, n_better < k))
            block_scores = [bs[n_better < k] for bs in block_scores]

            if len(candidates) <= k or i == self.max_n_blocks_eval:
                break

        if self.verbosity > 0:
            self.logger.warning(f"🏎️Racing: {len(candidates)} prompts remain after {i} blocks.")
        scores = np.concatenate(block_scores, axis=1).mean(axis=1)
        order = np.argsort(-scores)[:k]
        candidates = [candidates[i] for i in order]
        self.scores = scores[order]

        return candidates

    def optimize(self, n_steps: int) -> List[str]:
        """
        Main optimization loop that evolves the prompt population.

        Parameters:
            n_steps (int): Number of optimization steps to perform.

        Returns:
            List[str]: The final population of prompts after optimization.
        """
        for _ in range(n_steps):
            offsprings = self._crossover(self.prompt_objects)
            mutated = self._mutate(offsprings)
            combined = self.prompt_objects + mutated
            self.prompt_objects = self._do_racing(combined, self.population_size)
            self.prompts = [p.construct_prompt() for p in self.prompt_objects]
            continue_optimization = self._on_step_end()
            if not continue_optimization:
                break

        self._on_train_end()

        return self.prompts

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        state.pop("predictor", None)
        state.pop("logger", None)
        state.pop("meta_llm", None)
        state.pop("downstream_llm", None)

        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        self.predictor = None
        self.logger = getLogger(__name__)
