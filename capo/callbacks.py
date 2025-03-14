import os
from datetime import datetime

import dill
import pandas as pd
from promptolution.callbacks import Callback, CSVCallback


class PickleCallback(Callback):
    def __init__(self, output_dir, save_all_steps=False):
        self.output_dir = output_dir
        self.save_all_steps = save_all_steps
        self.count = 0

    def on_step_end(self, optimizer):
        self.count += 1
        if self.save_all_steps:
            with open(f"{self.output_dir}{self.count}.pickle", "wb") as f:
                dill.dump(optimizer, f)
        else:
            with open(f"{self.output_dir}optimizer.pickle", "wb") as f:
                dill.dump(optimizer, f)

        return True


class PromptScoreCallback(Callback):
    def __init__(self, dir, save_all_steps=False):
        """Initialize the PromptScoreCallback."""
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir
        self.save_all_steps = save_all_steps
        self.count = 0

    def on_step_end(self, optimizer):
        if hasattr(optimizer.task, "prompt_score_cache"):
            eval_dict = optimizer.task.prompt_score_cache

            prompts = set()
            block_ids = set()
            for prompt, block_id in eval_dict.keys():
                prompts.add(prompt)
                block_ids.add(block_id)

            prompts = sorted(list(prompts))
            block_ids = sorted(list(block_ids))

            df = pd.DataFrame(index=prompts, columns=block_ids, dtype=float)

            ordered_columns = [col for col, _ in optimizer.task.blocks if col in df.columns]
            df = df[ordered_columns]

            for (prompt, block_id), score in eval_dict.items():
                df.at[prompt, block_id] = score.mean()

            if self.save_all_steps:
                csv_path = os.path.join(self.dir, f"prompt_scores_{self.count}.csv")
                df.to_csv(csv_path)
                self.count += 1
            else:
                csv_path = os.path.join(self.dir, "prompt_scores.csv")
                df.to_csv(csv_path)

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
