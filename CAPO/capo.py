from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.llms.base_llm import BaseLLM
from promptolution.llms.api_llm import APILLM
from promptolution.tasks.base_task import BaseTask
from promptolution.tasks import ClassificationTask
from promptolution.predictors.base_predictor import BasePredictor
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples
from typing import List, Tuple, Callable, Dict
import random
import numpy as np
import pandas as pd

FEW_SHOT_TEMPLATE = """<instruction>

<examples>"""
# langchain has a class for few shot templates ... shall we use that?

DOWNSTREAM_TEMPLATE = """
{instruction}
Input: {input}
Output:
"""

CROSSOVER_TEMPLATE = """
Combine the following prompts:

Prompt 1: {mother}
Prompt 2: {father}

Return the result between <prompt> and </prompt>.
"""

MUTATION_TEMPLATE = """
Improve the prompt and return the result between <prompt> and </prompt>:

{instruction}
"""


class Prompt:
    def __init__(self, instruction: str, examples: List[Tuple[str, str]]):
        self.instruction = instruction
        self.examples = examples  # List of (input, output with reasoning)
        # block evaluation scores
        self.block_scores: Dict[int, float] = {}

    def construct_prompt(self) -> str:
        examples = "\n".join([f"Input: {x}\nOutput: {y}" for x, y in self.examples])
        return FEW_SHOT_TEMPLATE.replace("<instruction>", self.instruction).replace(
            "<examples>", examples
        )

    def evaluate_on_block(
        self,
        block: List[Tuple[str, str]],
        block_id: int,
        task: BaseTask,
        predictor: BasePredictor,
    ) -> float:
        if block_id in self.block_scores:
            return self.block_scores[block_id]

        temp_task = ClassificationTask.from_dataframe(
            pd.DataFrame(block, columns=["x", "y"]), description=task.description
        )
        prompt = self.construct_prompt()
        prompt_length = len(prompt.split(" "))
        score = temp_task.evaluate([prompt], predictor)
        self.block_scores[block_id] = score

        return score + prompt_length * 0.01

    def get_score(self) -> float:
        return np.mean(list(self.block_scores.values()))


class CAPOptimizer(BaseOptimizer):
    def __init__(
        self,
        initial_prompts: List[str],
        task: BaseTask,
        dataset: List[Tuple[str, str]],
        meta_llm: BaseLLM,
        downstream_llm: BaseLLM,
        block_size: int,
        crossovers_per_iter: int,
        upper_shots: int,
        max_n_blocks_eval: int,
        test_statistic: Callable,
        crossover_meta_prompt: str = None,
        mutation_meta_prompt: str = None,
        callbacks: List[Callable] = [],
        predictor=None,
    ):
        super().__init__(initial_prompts, task, callbacks, predictor)

        if isinstance(dataset, pd.DataFrame):
            dataset = [(x, y) for x, y in zip(dataset["x"], dataset["y"])]
        self.dataset = dataset

        self.meta_llm = meta_llm
        self.downstream_llm = downstream_llm

        if crossover_meta_prompt is None:
            crossover_meta_prompt = CROSSOVER_TEMPLATE
        self.crossover_meta_prompt = crossover_meta_prompt

        if mutation_meta_prompt is None:
            mutation_meta_prompt = MUTATION_TEMPLATE
        self.mutation_meta_prompt = mutation_meta_prompt

        self.population_size = len(initial_prompts)
        self.block_size = block_size
        self.crossovers_per_iter = crossovers_per_iter
        self.upper_shots = upper_shots
        self.max_n_blocks_eval = max_n_blocks_eval
        self.test_statistic = test_statistic
        self.blocks = self._split_into_blocks()
        self.population = self._initialize_population(initial_prompts)

    def _split_into_blocks(self) -> List[List[Tuple[str, str]]]:
        random.shuffle(self.dataset)
        blocks = [
            self.dataset[i : i + self.block_size]
            for i in range(0, len(self.dataset), self.block_size)
        ]
        return [(i, block) for i, block in enumerate(blocks)]

    def _initialize_population(self, initial_prompts: List[str]) -> List[Prompt]:
        population = []
        for instruction in initial_prompts:
            num_shots = random.randint(0, self.upper_shots)
            xi_samples = random.sample([x for x, _ in self.dataset], num_shots)
            examples = []
            for xi in xi_samples:
                theta = self.downstream_llm.get_response(
                    [f"{instruction}\nInput: {xi}\nOutput:"]
                )[0]
                examples.append((xi, theta))
            population.append(Prompt(instruction, examples))
        return population

    def _crossover(self, parents: List[Prompt]) -> List[Prompt]:
        offsprings = []
        for _ in range(self.crossovers_per_iter):
            mother_prompt, father_prompt = random.sample(parents, 2)
            # Combine instructions from both parents using meta-LLM
            crossover_prompt = (
                self.crossover_meta_prompt.replace(
                    "<mother>", mother_prompt.instruction
                )
                .replace("<father>", father_prompt.instruction)
                .strip()
            )
            child_instr = (
                self.meta_llm.get_response([crossover_prompt])[0]
                .split("<prompt>")[1]
                .split("</prompt>")[0]
                .strip()
            )
            # Combine examples from both parents
            child_examples = random.sample(
                mother_prompt.examples + father_prompt.examples,
                int(
                    len(mother_prompt.examples) * 0.5
                    + len(father_prompt.examples) * 0.5
                ),
            )

            offsprings.append(Prompt(child_instr, child_examples))
        return offsprings

    def _mutate(self, offsprings: List[Prompt]) -> List[Prompt]:
        mutated = []
        for prompt in offsprings:
            # Mutate instruction using meta-LLM
            mutation_prompt = self.mutation_meta_prompt.replace(
                "<instruction>", prompt.instruction
            )
            new_instr = (
                self.meta_llm.get_response([mutation_prompt])[0]
                .split("<prompt>")[-1]
                .split("</prompt>")[0]
                .strip()
            )
            # Sample number of shots for the new instruction
            num_shots = random.randint(0, self.upper_shots)
            num_new = random.randint(0, num_shots)
            num_old = num_shots - num_new
            # Sample new examples
            new_examples = random.sample([x for x, _ in self.dataset], num_new)
            for example in new_examples:
                reasoning = self.downstream_llm.get_response(
                    [
                        DOWNSTREAM_TEMPLATE.replace(
                            "<instruction>", "mutation_prompt"
                        ).replace("<example>", example)
                    ],
                    return_seq=True,
                )[0]
                # TODO: check if answer is correct!
                new_examples.append((example, reasoning))
            # Sample old examples
            old_examples = random.sample(
                prompt.examples, min(num_old, len(prompt.examples))
            )
            combined = old_examples + new_examples
            random.shuffle(combined)
            mutated.append(Prompt(new_instr, combined))
        return mutated

    def _do_racing(self, candidates: List[Prompt], k: int) -> List[Prompt]:
        random.shuffle(self.blocks)
        for i, (block_id, block) in enumerate(self.blocks):
            # TODO parralelize for speed up!!
            for prompt in candidates:
                prompt.evaluate_on_block(block, block_id, self.task, self.predictor)
            # Calculate average scores and eliminate
            for candidate in candidates:
                score = candidate.get_score()
                # count the number of prompts that perform significantly (using test_statistic) better than prompt
                n_better = sum(
                    [
                        self.test_statistic(score, other.get_score())
                        for other in candidates
                    ]
                )
                if n_better >= k:
                    candidates.remove(candidate)
            if len(candidates) <= k or i == self.max_n_blocks_eval:
                break
        return candidates[:k]

    def optimize(self, n_steps: int) -> List[str]:
        for step in range(n_steps):
            print([p.construct_prompt() for p in self.population])

            offsprings = self._crossover(self.population)
            mutated = self._mutate(offsprings)
            combined = self.population + mutated
            self.population = self._do_racing(combined, self.population_size)
            self._on_step_end()
        self._on_train_end()
        return self.population


# Example usage
if __name__ == "__main__":
    token = open("deepinfratoken.txt", "r").read()

    meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
    downstream_llm = meta_llm
    # Mock dataset and parameters
    df = pd.read_csv(
        "hf://datasets/nakamoto-yama/dt-mappings/yama_dt_mappings.csv"
    ).rename({"Degree Type": "x", "Mapping": "y"}, axis=1)
    task = ClassificationTask.from_dataframe(df, description="test test")
    initial_prompts = [
        create_prompts_from_samples(task, downstream_llm) for _ in range(10)
    ]

    predictor = Classificator(downstream_llm, df["y"].unique())
    test_statistic = lambda x, y: x > y  # simple test statistic for now

    capo = CAPOptimizer(
        initial_prompts=initial_prompts,
        task=task,
        dataset=df,
        meta_llm=meta_llm,
        downstream_llm=downstream_llm,
        block_size=30,
        crossovers_per_iter=2,
        upper_shots=5,
        test_statistic=test_statistic,
        max_n_blocks_eval=5,
        predictor=predictor,
    )
    best_prompt = capo.optimize(n_steps=3)
    print(f"Best instruction: \n\n{best_prompt}")
