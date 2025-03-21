import os
from glob import glob

from capo.configs.config_to_command import generate_experiment_command, generate_individual_configs
from capo.configs.experiment_configs import BENCHMARK_CONFIG

if __name__ == "__main__":
    individual_configs = generate_individual_configs(BENCHMARK_CONFIG)

    c = 0
    for config in individual_configs:
        # check if config.output_dir already exists, if so, skip
        complete_path = glob(config.output_dir + "**", recursive=True)
        if os.path.exists(config.output_dir) and not any(
            ["step_results_eval.csv" in c for c in complete_path]
        ):
            command = generate_experiment_command(config, evaluate=True)
        elif not os.path.exists(config.output_dir):
            command = generate_experiment_command(
                config, time="0-02:00:00", gres="gpu:1", partition="mcml-hgx-a100-80x4"
            )
        else:
            continue
        print(command)
        c += 1

    print(f"Total number of jobs: {c}")
