from typing import NoReturn, Dict, Optional
import json
import os
import subprocess
import argparse

def parse_endpoints(path_to_endpoints_file: str, ) -> Dict[str, str]:
    with open(path_to_endpoints_file, "r+") as file:
        endpoints = json.load(file)
    return endpoints

def run_smoke_tests(
        endpoints: Dict[str, str],
        endpoint_type: str, 
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
        endpoint_type: str,
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
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument("--benchmark_repo", type=str)
    parser.add_argument("--num_fewshot", type=int)
    parser.add_argument("--write_out_base", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--endpoint_type", type=str, default="dev")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if args.run_tests and not args.tests_file:
        raise RuntimeError("Please specify path to file with tests to run")
    if args.run_benchmark and not args.benchmark_repo:
        raise RuntimeError("Please specify path to repository with benchmark source (works with lm-evaluation-harness)")
    if args.endpoint_type not in ["dev", "prod"]:
        raise RuntimeError("Please specify only dev or prod type of endpoints")

    endpoints = parse_endpoints(args.endpoints_file)
    
    if args.run_tests:
        run_smoke_tests(
            endpoints, 
            args.endpoint_type,
            args.tests_file, 
            write_out_base_path=args.write_out_base,
        )
    else:
        run_benchmark(
            endpoints,
            args.endpoint_type,
            args.benchmark_repo,
            num_fewshot=args.num_fewshot,
            write_out_base_path=args.write_out_base,
            task=args.task
        )
if __name__ == "__main__":
    main()
