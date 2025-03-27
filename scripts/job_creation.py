import argparse
import os
from glob import glob

from capo.configs.config_to_command import generate_command, generate_individual_configs
from capo.configs.experiment_configs import ABLATION_CONFIG, BENCHMARK_CONFIG, HYPERPARAMETER_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", default=None)
parser.add_argument("--no-evals", action="store_true")
parser.add_argument("--run-ablations", action="store_true")
parser.add_argument("--run-hp", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    if args.run_ablations:
        config = ABLATION_CONFIG
    elif args.run_hp:
        config = HYPERPARAMETER_CONFIG
    else:
        config = BENCHMARK_CONFIG
    individual_configs = generate_individual_configs(config)
    if args.optimizer is not None:
        individual_configs = [
            c for c in individual_configs if c.optimizers[0].name == args.optimizer
        ]
    c = 0
    for config in individual_configs:
        complete_path = glob(config.output_dir + "**", recursive=True)
        if not os.path.exists(config.output_dir) or not any(
            ["step_results.parquet" in c for c in complete_path]
        ):
            command = generate_command(
                config, time="0-02:30:00", gres="gpu:1", partition="mcml-hgx-a100-80x4"
            )
        elif (
            os.path.exists(config.output_dir)
            and not any(["step_results_eval.csv" in c for c in complete_path])
            and not args.no_evals
        ):
            dirs = [c for c in complete_path if "step_results.parquet" in c]
            config.output_dir = dirs[0].replace("step_results.parquet", "").replace("\\", "/")
            command = generate_command(
                config,
                evaluate=True,
                time="0-05:30:00",
                gres="gpu:1",
                partition="mcml-hgx-a100-80x4",
            )
        else:
            continue
        print(command)
        c += 1

    print(f"Total number of jobs: {c}")
