from typing import NoReturn, Dict, Optional

import os
import subprocess
import argparse

def parse_endpoints(
        path_to_endpoints_file: str, 
        sep: Optional[str] = None, 
        remove_prefix: Optional[str] = None
    ) -> Dict[str, str]:
    endpoints = dict()
    with open(path_to_endpoints_file) as file:
        for line in file.readlines():
            splitted_line = line.split(sep=sep)
            for i in range(len(splitted_line)):
                splitted_line[i] = splitted_line[i].strip()
                if remove_prefix and splitted_line[i].startswith(remove_prefix):
                    splitted_line[i] = splitted_line[i][len(remove_prefix):]
            endpoints[splitted_line[0]] = splitted_line[1]
    return endpoints

def run_smoke_tests(
        endpoints: Dict[str, str],
        path_to_tests_file: str, 
        write_out_base_path: str = "./"
    ) -> NoReturn:
    if not os.environ.get("OCTOAI_TOKEN"):
        os.environ["OCTOAI_TOKEN"] = os.environ.get("OCTOI_API_KEY")
    for endpoint_name in endpoints.keys():
        print(f"--------------------------------------------------------------------------")
        print(f"Running smoke_tests for {endpoint_name}")
        print(f"--------------------------------------------------------------------------")
        subprocess.run(
            f"tmux new-session -d -s {endpoint_name} ",
            shell=True, 
            universal_newlines=True
        )
        subprocess.run(
            f"tmux send-keys -t {endpoint_name} "
            f"\"ENDPOINT={endpoints[endpoint_name]}/{endpoint_name} "
            f"python3 -m pytest {path_to_tests_file} " 
            f"--model_name={endpoint_name} > {os.path.join(write_out_base_path, f'test_{endpoint_name}.log')}\" Enter",
            shell=True,
            universal_newlines=True
        )
        subprocess.run(
            f"tmux capture-pane -t {endpoint_name} -p",
            shell=True, 
            universal_newlines=True
        )

def run_benchmark(
        endpoints: Dict[str, str],
        path_to_benchmark_repo: str,
        num_fewshot: int = 0,
        write_out_base_path: str = "./",
        task: str = "gsm8k",
        
    ) -> NoReturn:
    if not os.environ.get("OCTOAI_API_KEY"):
        os.environ["OCTOAI_API_KEY"] = os.environ.get("OCTOAI_TOKEN")
    for endpoint_name in endpoints.keys():
        if not os.path.exists(os.path.join(write_out_base_path, endpoint_name)):
            os.makedirs(os.path.join(write_out_base_path, endpoint_name))
        work_dir = os.getcwd()
        os.chdir(os.path.join(write_out_base_path, endpoint_name))

        if not os.path.exists(os.path.join(write_out_base_path, endpoint_name,f"nf{num_fewshot}")):
            os.makedirs(os.path.join(write_out_base_path, endpoint_name, f"nf{num_fewshot}"))

        print(f"--------------------------------------------------------------------------")
        print(f"Running benchmark for {endpoint_name} with --num_fewshot={num_fewshot}")
        print(f"--------------------------------------------------------------------------")

        subprocess.run(
            f"tmux new-session -d -s {endpoint_name}-nf{num_fewshot} ",
            shell=True, 
            universal_newlines=True
        )
        subprocess.run(
            f"tmux send-keys -t {endpoint_name}-nf{num_fewshot} "
            f"\"python3 {path_to_benchmark_repo}/main.py "
            f"--model=octoai "
            f"--model_args=\'model_name={endpoint_name}\' "
            f"--task={task} "
            f"--output_path=./nf{num_fewshot}/res_greedy_nf{num_fewshot}_{task}_{endpoint_name}.json " 
            f"--no_cache " 
            f"--num_fewshot={num_fewshot} " 
            f"--batch_size=1 " 
            f"--write_out " 
            f"--output_base_path=./nf{num_fewshot}/\" Enter", 
            shell=True, 
            universal_newlines=True
        )
        subprocess.run(
            f"tmux capture-pane -t {endpoint_name}-nf{num_fewshot} -p",
            shell=True, 
            universal_newlines=True
        )
        os.chdir(work_dir)
    print("Done")

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", required=True, type=str)
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument("--tests_file", type=str)
    parser.add_argument("--remove_prefix", type=str)
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument("--benchmark_repo", type=str)
    parser.add_argument("--num_fewshot", type=int)
    parser.add_argument("--write_out_base", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if args.run_tests and args.run_benchmark:
        raise RuntimeError("Please specify either running tests or running benchmark")
    if args.run_tests and not args.tests_file:
        raise RuntimeError("Please specify path to file with tests to run")
    if args.run_benchmark and not args.benchmark_repo:
        raise RuntimeError("Please specify path to repository with benchmark source (works with lm-evaluation-harness)")

    endpoints = parse_endpoints(args.endpoints_file, remove_prefix=args.remove_prefix)
    
    if args.run_tests:
        run_smoke_tests(
            endpoints, 
            args.tests_file, 
            write_out_base_path=args.write_out_base
        )
    else:
        run_benchmark(
            endpoints,
            args.benchmark_repo,
            num_fewshot=args.num_fewshot,
            write_out_base_path=args.write_out_base,
            task=args.task
        )
if __name__ == "__main__":
    main()
