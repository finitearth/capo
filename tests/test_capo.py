import random  # noqa: F401
from unittest.mock import Mock, patch

import numpy as np
import pytest

from capo.capo import CAPOptimizer, Prompt


def mock_test_statistic(score_a, score_b, n_evals=1):
    return np.mean(score_a) > np.mean(score_b)


@pytest.fixture
def optimizer():
    task = Mock(
        spec=["description", "xs", "ys", "few_shots", "keys", "evaluate", "evaluate_on_block"]
    )
    task.description = "Test task"
    task.xs = ["input1", "input2", "input3"]
    task.ys = ["output1", "output2", "output3"]
    task.few_shots = [0, 1, 2]
    task.blocks = [(0, "block0"), (1, "block1")]
    task.evaluate = Mock(return_value=[0.5, 0.5])
    task.evaluate_on_block = Mock(return_value=np.array([[1] * 8 + [0] * 2, [1] * 5 + [0] * 5]))

    meta_llm = Mock()
    downstream_llm = Mock()
    predictor = Mock()
    predictor.predict.return_value = (["output"], ["Input: input\nOutput: output"])

    initial_prompts = ["Prompt 1", "Prompt 2"]

    with patch("capo.capo.CAPOTask.from_task", return_value=task):
        opt = CAPOptimizer(
            initial_prompts=initial_prompts,
            task=task,
            meta_llm=meta_llm,
            downstream_llm=downstream_llm,
            length_penalty=0.1,
            block_size=2,
            crossovers_per_iter=1,
            upper_shots=2,
            p_few_shot_reasoning=0.5,
            max_n_blocks_eval=2,
            few_shot_split_size=0.5,
            test_statistic=mock_test_statistic,
            shuffle_blocks_per_iter=False,
            predictor=predictor,
        )
    return opt


def test_initialize_population(optimizer):
    with patch("random.randint", return_value=1):
        population = optimizer._initialize_population(["Prompt 1", "Prompt 2"])
    assert len(population) == 2
    for prompt in population:
        assert prompt.instruction_text in ["Prompt 1", "Prompt 2"]
        assert len(prompt.few_shots) == 1
        assert isinstance(prompt.few_shots[0], str)


def test_create_few_shot_examples(optimizer):
    instruction = "Test instruction"
    optimizer.predictor.predict.return_value = (["output1"], ["Input: input1\nOutput: output1"])

    with patch("random.random", return_value=0.2), patch("random.sample", return_value=[0]):
        few_shots = optimizer._create_few_shot_examples(instruction, 1)
    assert len(few_shots) == 1
    assert few_shots[0] == "Input: input1\nOutput: output1"

    with patch("random.random", return_value=0.8), patch("random.sample", return_value=[0]):
        few_shots = optimizer._create_few_shot_examples(instruction, 1)
    assert few_shots[0] == "Input: input1\nOutput: output1"


def test_crossover(optimizer):
    parents = [Prompt("Parent 1", ["shot1"]), Prompt("Parent 2", ["shot2"])]
    optimizer.meta_llm.get_response.return_value = ["<prompt>Child 1</prompt>"]

    with patch("random.sample", side_effect=[parents, ["shot1"]]):
        offsprings = optimizer._crossover(parents)
    assert len(offsprings) == 1
    assert offsprings[0].instruction_text == "Child 1"
    assert offsprings[0].few_shots == ["shot1"]


def test_mutate(optimizer):
    offspring = Prompt("Offspring", ["shot1"])
    optimizer.meta_llm.get_response.return_value = ["<prompt>Mutated</prompt>"]

    with patch("random.randint", side_effect=[1, 0]), patch(
        "random.sample", return_value=["shot1"]
    ):
        with patch.object(optimizer, "_create_few_shot_examples", return_value=[]):
            mutated = optimizer._mutate([offspring])
    assert len(mutated) == 1
    assert mutated[0].instruction_text == "Mutated"
    assert len(mutated[0].few_shots) == 1


def test_do_racing_with_guaranteed_elimination(optimizer):
    candidates = [Prompt("P1 - Champion", []), Prompt("P2 - Middle", []), Prompt("P3 - Loser", [])]

    optimizer.task.evaluate_on_block.side_effect = [
        np.array(
            [
                [1] * 9 + [0] * 1,  # P1: High scores
                [1] * 5 + [0] * 5,  # P2: Middle scores
                [1] * 1 + [0] * 9,  # P3: Very low scores
            ]
        )
    ]

    optimizer.max_prompt_length = 10
    for c in candidates:
        c.construct_prompt = Mock(return_value="short prompt")

    # Debug to confirm mock setup
    survivors = optimizer._do_racing(candidates, k=2)
    assert len(survivors) == 2
    survivor_texts = [s.instruction_text for s in survivors]
    print("Survivors:", survivor_texts)
    assert "P1 - Champion" in survivor_texts, "Best prompt P1 should not be eliminated"
    assert "P3 - Loser" not in survivor_texts, "Worst prompt P3 should be eliminated"
    assert "P2 - Middle" in survivor_texts, "Middle prompt P2 should survive"


def test_optimize(optimizer):
    optimizer.meta_llm.get_response.side_effect = [
        ["<prompt>Child</prompt>"],
        ["<prompt>Mutated</prompt>"],
    ]
    optimizer.task.evaluate_on_block.return_value = np.array(
        [[1] * 8 + [0] * 2, [1] * 5 + [0] * 5, [1] * 2 + [0] * 8]
    )

    # Extended side_effect to cover all random.sample calls
    sample_side_effect = [optimizer.prompts, [0], [0], [], []]  # 5 values for safety
    with patch("random.sample", side_effect=sample_side_effect):
        result = optimizer.optimize(n_steps=1)
    assert len(result) == 2
    assert all(isinstance(p, str) for p in result)
