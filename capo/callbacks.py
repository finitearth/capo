import json
import os
from datetime import datetime

import dill
import numpy as np
import pandas as pd
from promptolution.callbacks import Callback, CSVCallback


class PickleCallback(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.count = 0

    def on_step_end(self, optimizer):
        self.count += 1
        with open(f"{self.output_dir}{self.count}.pickle", "wb") as f:
            dill.dump(optimizer, f)

        return True


class NumberOfEvalsCallback(Callback):
    def __init__(self, dir):
        self.dir = dir

    def on_train_end(self, optimizer):
        if hasattr(optimizer.task, "prompt_score_cache"):
            eval_dict = optimizer.task.prompt_score_cache  # (prompt, block_id): score
            with open(f"{self.dir}/block_evals.json", "w") as f:
                json.dump(eval_dict, f)
        else:
            print("Task has no prompt_score_cache attribute.")
        return True


class CSVCallback(CSVCallback):
    def __init__(self, dir):
        """Initialize the CSVCallback.

        Args:
        dir (str): Directory the CSV file is saved to.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.dir = dir
        self.step = 0
        self.input_tokens_meta = 0
        self.output_tokens_meta = 0
        self.input_tokens_downstream = 0
        self.output_tokens_downstream = 0
        self.start_time = datetime.now()
        self.step_time = datetime.now()

    def on_step_end(self, optimizer):
        """Save prompts and scores to csv.

        Args:
        optimizer: The optimizer object that called the callback
        """
        self.step += 1
        df = pd.DataFrame(
            {
                "step": [self.step] * len(optimizer.prompts),
                "input_tokens_meta_llm": [
                    optimizer.meta_llm.input_token_count - self.input_tokens_meta
                ]
                * len(optimizer.prompts),
                "output_tokens_meta_llm": [
                    optimizer.meta_llm.output_token_count - self.output_tokens_meta
                ]
                * len(optimizer.prompts),
                "input_tokens_downstream_llm": [
                    optimizer.downstream_llm.input_token_count - self.input_tokens_downstream
                ]
                * len(optimizer.prompts),
                "output_tokens_downstream_llm": [
                    optimizer.downstream_llm.output_token_count - self.output_tokens_downstream
                ]
                * len(optimizer.prompts),
                "time_elapsed": [(datetime.now() - self.step_time).total_seconds()]
                * len(optimizer.prompts),
                "score": optimizer.scores,
                "prompt": optimizer.prompts,
            }
        )
        self.step_time = datetime.now()
        self.input_tokens_meta = optimizer.meta_llm.input_token_count
        self.output_tokens_meta = optimizer.meta_llm.output_token_count
        self.input_tokens_downstream = optimizer.downstream_llm.input_token_count
        self.output_tokens_downstream = optimizer.downstream_llm.output_token_count

        if not os.path.exists(self.dir + "step_results.csv"):
            df.to_csv(self.dir + "step_results.csv", index=False)
        else:
            df.to_csv(self.dir + "step_results.csv", mode="a", header=False, index=False)

        return True

    def on_train_end(self, optimizer):
        """Called at the end of training.

        Args:
        optimizer: The optimizer object that called the callback.
        """
        df = pd.DataFrame(
            dict(
                steps=self.step,
                input_tokens_llm=optimizer.meta_llm.input_token_count,
                output_tokens=optimizer.meta_llm.output_token_count,
                time_elapsed=(datetime.now() - self.start_time).total_seconds(),
                end_time=datetime.now(),
                start_time=self.start_time,
                avg_score=np.array(optimizer.scores).mean(),
                end_prompts=str(optimizer.prompts),
            ),
            index=[0],
        )

        if not os.path.exists(self.dir + "train_results.csv"):
            df.to_csv(self.dir + "train_results.csv", index=False)
        else:
            df.to_csv(self.dir + "train_results.csv", mode="a", header=False, index=False)

        return True
