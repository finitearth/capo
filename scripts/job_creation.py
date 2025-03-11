from capo.configs.config_to_command import generate_command, generate_individual_configs
from capo.configs.experiment_configs import BENCHMARK_CONFIG

if __name__ == "__main__":
    individual_configs = generate_individual_configs(BENCHMARK_CONFIG)

    print(len(individual_configs))

    for config in individual_configs:
        command = generate_command(config, time="00:30:00")
        print(command)
