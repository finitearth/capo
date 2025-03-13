import json
import os
from datetime import datetime

import dill
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
        data = {
            "step": [self.step] * len(optimizer.prompts),
            "timestamp": [datetime.now()] * len(optimizer.prompts),
            "time_elapsed": [(datetime.now() - self.step_time).total_seconds()]
            * len(optimizer.prompts),
            "score": optimizer.scores,
            "prompt": optimizer.prompts,
        }
        if hasattr(optimizer, "meta_llm"):
            data["input_tokens_meta_llm"] = [
                optimizer.meta_llm.input_token_count - self.input_tokens_meta
            ] * len(optimizer.prompts)
            data["output_tokens_meta_llm"] = [
                optimizer.meta_llm.output_token_count - self.output_tokens_meta
            ] * len(optimizer.prompts)
            self.input_tokens_meta = optimizer.meta_llm.input_token_count
            self.output_tokens_meta = optimizer.meta_llm.output_token_count

        if hasattr(optimizer, "downstream_llm"):
            data["input_tokens_downstream_llm"] = [
                optimizer.downstream_llm.input_token_count - self.input_tokens_downstream
            ] * len(optimizer.prompts)
            data["output_tokens_downstream_llm"] = [
                optimizer.downstream_llm.output_token_count - self.output_tokens_downstream
            ] * len(optimizer.prompts)
            self.input_tokens_downstream = optimizer.downstream_llm.input_token_count
            self.output_tokens_downstream = optimizer.downstream_llm.output_token_count

        df = pd.DataFrame(data)
        self.step_time = datetime.now()

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
        return True
