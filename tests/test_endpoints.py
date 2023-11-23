from typing import NoReturn, Dict, List
from datetime import date
from pathlib import Path
from utils import init_gspread_client
import json
import os
import re
import subprocess
import argparse

import gspread

TASKS = ["gsm8k", "truthfulqa_gen", "triviaqa"]

def parse_endpoints(path_to_endpoints_file: str, ) -> Dict[str, str]:
    with open(path_to_endpoints_file, "r+") as file:
        endpoints = json.load(file)
    return endpoints

def run_smoke_tests(
        endpoints: Dict[str, str],
        endpoint_type: str, 
        path_to_tests_file: str, 
        write_out_base_path: str = "./",
        write_table: bool = True,
    ) -> None:
    if not os.environ.get("OCTOAI_TOKEN"):
        os.environ["OCTOAI_TOKEN"] = os.environ.get("OCTOI_API_KEY")

    for col_num, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        print(f"--------------------------------------------------------------------------")
        print(f"Running smoke_tests for {model_name}")
        print(f"--------------------------------------------------------------------------")
        subprocess.run(
            f"tmux new-session -d -s {model_name} ",
            shell=True, 
            universal_newlines=True
        )
        log_file = os.path.join(write_out_base_path, f'test_{model_name}.log')
        write_table_command = ""

        if write_table:
            write_table_command = f"python {os.path.join(str(Path(__file__).parent), 'process_logs.py')} --path_to_log={log_file} --col_num={col_num} --model_name={model_name}"

        subprocess.run(
            f"tmux send-keys -t {model_name} "
            f"\"ENDPOINT={endpoint['url']} "
            f"python3 -m pytest {path_to_tests_file} " 
            f"--model_name={model_name} > {log_file}\" Enter",
            shell=True,
            universal_newlines=True
        )

        subprocess.run(
            f"tmux send-keys -t {model_name} "
            f"\"{write_table_command}\" Enter", 
            shell=True, 
            universal_newlines=True
        )
    
    print("Done")

def run_benchmark(
        endpoints: Dict[str, str],
        endpoint_type: str,
        path_to_benchmark_repo: str,
        num_fewshot: int = 0,
        write_out_base_path: str = "./",
        task: str = "gsm8k",
        
    ) -> None:
    if not os.environ.get("OCTOAI_API_KEY"):
        os.environ["OCTOAI_API_KEY"] = os.environ.get("OCTOAI_TOKEN")
    for endpoint in endpoints[endpoint_type]:
        if not os.path.exists(os.path.join(write_out_base_path, endpoint["model"])):
            os.makedirs(os.path.join(write_out_base_path, endpoint["model"]))
        work_dir = os.getcwd()
        os.chdir(os.path.join(write_out_base_path, endpoint["model"]))

        if not os.path.exists(os.path.join(write_out_base_path, endpoint["model"],f"nf{num_fewshot}")):
            os.makedirs(os.path.join(write_out_base_path, endpoint["model"], f"nf{num_fewshot}"))

        print(f"--------------------------------------------------------------------------")
        print(f"Running benchmark for {endpoint['model']} with --num_fewshot={num_fewshot}")
        print(f"--------------------------------------------------------------------------")

        subprocess.run(
            f"tmux new-session -d -s {endpoint['model']}-nf{num_fewshot} ",
            shell=True, 
            universal_newlines=True
        )
        subprocess.run(
            f"tmux send-keys -t {endpoint['model']}-nf{num_fewshot} "
            f"\"python3 {path_to_benchmark_repo}/main.py "
            f"--model=octoai "
            f"--model_args=\'model_name={endpoint['model']}\' "
            f"--task={task} "
            f"--output_path=./nf{num_fewshot}/res_greedy_nf{num_fewshot}_{task}_{endpoint['model']}.json " 
            f"--no_cache " 
            f"--num_fewshot={num_fewshot} " 
            f"--batch_size=1 " 
            f"--write_out " 
            f"--output_base_path=./nf{num_fewshot}/\" Enter", 
            shell=True, 
            universal_newlines=True
        )
        
        subprocess.run(
            f"tmux capture-pane -t {endpoint['model']}-nf{num_fewshot} -p",
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
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--write_out_base", type=str, default="./")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--write_table", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if args.run_tests and not args.tests_file:
        raise RuntimeError("Please specify path to file with tests to run")
    if args.run_benchmark and not args.benchmark_repo:
        raise RuntimeError("Please specify path to repository with benchmark source (works with lm-evaluation-harness)")
    if args.endpoint_type not in ["dev", "prod"]:
        raise RuntimeError("Please specify only dev or prod type of endpoints")
    if args.run_all and not (args.test_file and args.benchmark_repo):
        raise RuntimeError("Please specify both test_file path and path to your local benchmark repository if you want to run all")

    endpoints = parse_endpoints(args.endpoints_file)

    if args.write_table:
        spreadsheet = init_gspread_client()
        pytest_nodes = subprocess.check_output(
            f"ENDPOINT=test pytest {args.tests_file} --collect-only",
            shell=True,
            text=True
        )
        test_names = []
        for line in pytest_nodes.split('\n'):
            match = re.search(r"test_[a-zA-Z_]+", line)
            if match:
                test_names.append(match[0])

        today = str(date.today())
        try:
            worksheet = spreadsheet.add_worksheet(title=today, rows=250, cols=100)
        except:
            worksheet = spreadsheet.worksheet(today)
        splitted_test_names = [[test_name] for test_name in test_names]
        worksheet.update(f"A2:A{2 + len(test_names)}", splitted_test_names)

    if args.run_all:
        run_smoke_tests(
            endpoints, 
            args.endpoint_type,
            args.tests_file, 
            write_out_base_path=args.write_out_base,
            write_table=args.write_table
        )
        
        for task in TASKS:
            run_benchmark(
                endpoints,
                args.endpoint_type,
                args.benchmark_repo,
                num_fewshot=args.num_fewshot,
                write_out_base_path=args.write_out_base,
                task=task
            )
    
    else:
        if args.run_tests:
            run_smoke_tests(
                endpoints, 
                args.endpoint_type,
                args.tests_file, 
                write_out_base_path=args.write_out_base,
                write_table=args.write_table
            )
        if args.run_benchmark:
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
