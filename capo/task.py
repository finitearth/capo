"""
Defines the Task class specific to the CAPO algorithm implementation.
Encapsulates task-specific functionality including evaluation metrics, dataset handling, and performance tracking for prompt optimization.
"""
import random

import numpy as np
from promptolution.tasks import ClassificationTask


class CAPOClassificationTask(ClassificationTask):
    def __init__(self, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.blocks = self._split_into_blocks()
        self.prompt_score_cache = {}  # (prompt, block_id): score

    def _split_into_blocks(self):
        num_samples = len(self.xs)
        indices = list(range(num_samples))
        random.shuffle(indices)

        blocks = [
            indices[i * self.block_size : (i + 1) * self.block_size]
            for i in range(num_samples // self.block_size)
        ]

        blocks = list(enumerate(blocks))

        return blocks

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

    def get_avg_scores(self, prompts):
        """Get the average scores for each prompt across all blocks.

        Args:
            prompts (List[str]): List of prompts to get scores for.

        Returns:
            List[float]: List of average scores for each prompt
        """
        prompt_scores = []
        for prompt in prompts:
            scores = []
            for block_id, _ in self.blocks:
                score = self.prompt_score_cache.get((prompt, block_id))
                if score is not None:
                    scores.append(score)
            prompt_scores.append(np.mean(scores))
        return np.array(prompt_scores)
