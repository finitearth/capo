import os

from capo.configs.config_to_command import generate_command, generate_individual_configs
from capo.configs.experiment_configs import BENCHMARK_CONFIG

if __name__ == "__main__":
    individual_configs = generate_individual_configs(BENCHMARK_CONFIG)

    c = 0
    for config in individual_configs:
        # check if config.output_dir already exists, if so, skip
        if os.path.exists(config.output_dir):
            continue
        command = generate_command(
            config, time="0-02:00:00", gres="gpu:1", partition="mcml-hgx-a100-80x4"
        )
        print(command)
        c += 1

    print(f"Total number of jobs: {c}")
