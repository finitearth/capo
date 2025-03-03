import random

import numpy as np
from promptolution.tasks import BaseTask, ClassificationTask


class CAPOTask(BaseTask):
    @classmethod
    def from_task(cls, task, few_shot_split_size=0.2, block_size=30):
        instance = cls.__new__(cls)
        instance.__dict__.update(task.__dict__)
        instance.few_shot_split_size = few_shot_split_size
        instance.block_size = block_size
        instance.prompt_score_cache = {}

        instance.ys = np.array([task.classes[y] for y in task.ys])
        instance.blocks, instance.few_shots = instance._split_into_blocks()

        return instance

    def __init__(self, block_size, few_shot_split_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.few_shot_split_size = few_shot_split_size
        self.blocks, self.few_shots = self._split_into_blocks()
        self.prompt_score_cache = {}  # (prompt, block_id): score

    def _split_into_blocks(self):
        num_samples = len(self.xs)
        indices = list(range(num_samples))
        random.shuffle(indices)

        n_few_shots = int(self.few_shot_split_size * num_samples)
        few_shots = indices[:n_few_shots]
        blocks = [
            indices[i : i + self.block_size]
            for i in range(n_few_shots, num_samples, self.block_size)
        ]

        blocks = list(enumerate(blocks))

        return blocks, few_shots

    def evaluate_on_block(self, prompts, block_id, predictor):
        _, block = self.blocks[block_id]

        xs = [self.xs[i] for i in block]
        ys = [self.ys[i] for i in block]

        # for each prompt, check if it has been evaluated: if not, append
        # to the list of prompts to evaluate
        # if yes, use the cached score
        to_be_evaluated = [
            prompt for prompt in prompts if (prompt, block_id) not in self.prompt_score_cache
        ]
        preds = predictor.predict(to_be_evaluated, xs)  # shape: P x N
        for prompt, pred in zip(to_be_evaluated, preds):
            score = np.array([self.metric([y], [p]) for y, p in zip(ys, pred)])
            self.prompt_score_cache[(prompt, block_id)] = score

        scores = [self.prompt_score_cache[(prompt, block_id)] for prompt in prompts]
        scores = np.array(scores)

        return scores


class CAPOClassificationTask(ClassificationTask, CAPOTask):
    pass
