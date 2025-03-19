import argparse
from glue.promptopt.instantiate import GluePromptOpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments needed by prompt manager")
    parser.add_argument('--llm_config_path', default=None)
    parser.add_argument('--prompt_config_path', default=None)
    parser.add_argument('--setup_config_path', default=None)
    parser.add_argument('--train_file_name', default=None)
    parser.add_argument('--test_file_name', default=None)
    parser.add_argument('--dataset_processor_pkl_path', default=None)
    parser.add_argument('--prompt_pool_path', default=None)

    args = parser.parse_args()

    gp = GluePromptOpt(args.llm_config_path,
                       args.promptopt_config_path,
                       args.setup_config_path,
                       args.train_file_name,
                       args.dataset_processor_pkl_path,
                       args.prompt_pool_path)

    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    if args.test_file_name:
        accuracy = gp.evaluate(args.test_file_name)
        print(f"accuracy: {accuracy}")

