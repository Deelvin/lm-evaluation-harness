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

FEWSHOTS_PER_TASK = {"gsm8k": [0, 5, 8], "truthfulqa_gen": [0], "triviaqa": [0, 5]}

def parse_endpoints(path_to_endpoints_file: str, ) -> Dict[str, str]:
    with open(path_to_endpoints_file, "r+") as file:
        endpoints = json.load(file)
    return endpoints

def run_benchmark(
        endpoints: Dict[str, str],
        endpoint_type: str,
        path_to_benchmark_repo: str,
        num_fewshot: int = 0,
        write_out_base_path: str = "./",
        task: str = "gsm8k",
        write_table: bool = True,
        debug: bool = False,
        limit: int = 3,

    ) -> None:
    if not os.environ.get("OCTOAI_API_KEY"):
        os.environ["OCTOAI_API_KEY"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}")

    current_session = 0
    for col_num, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        res_path = os.path.join(write_out_base_path, model_name)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        work_dir = os.getcwd()
        os.chdir(res_path)

        if not os.path.exists(os.path.join(write_out_base_path, endpoint["model"],f"nf{num_fewshot}")):
            os.makedirs(os.path.join(write_out_base_path, endpoint["model"], f"nf{num_fewshot}"))

        print(f"-----------------------------------------------------------------------------------")
        print(f"Running benchmark for {endpoint_type}_{model_name} with --num_fewshot={num_fewshot}")
        print(f"-----------------------------------------------------------------------------------")
        print(f"Results from this run will be saved in the following path: {res_path}")

        if current_session < limit:
            subprocess.run(
                f"tmux new-session -d -s {current_session} ",
                shell=True, 
                universal_newlines=True
            )

        write_table_command = ""

        res_output = os.path.join(
            res_path, 
            f"nf{num_fewshot}/res_greedy_nf{num_fewshot}_{task}_{endpoint_type}_{model_name}.json"
        )

        if write_table:
            write_table_command = f"python {os.path.join(str(Path(__file__).parent), 'process_logs.py')} --path_to_results={res_output} --model_name={endpoint_type}_{model_name}"

        subprocess.run(
            f"tmux send-keys -t {current_session % limit} "
            f"\"python3 {path_to_benchmark_repo}/main.py "
            f"--model=octoai "
            f"--model_args=\'model_name={model_name}\' "
            f"--task={task} "
            f"--output_path={res_output} " 
            f"--no_cache " 
            f"--num_fewshot={num_fewshot} " 
            f"--batch_size=1 " 
            f"--write_out " 
            f"--limit=16 "
            f"--output_base_path=./nf{num_fewshot}/\" Enter", 
            shell=True, 
            universal_newlines=True
        )

        subprocess.run(
            f"tmux send-keys -t {current_session % limit} "
            f"\"{write_table_command}\" Enter", 
            shell=True, 
            universal_newlines=True
        )
        
        os.chdir(work_dir)

    for num_session in range(limit):
        subprocess.run(
            f"tmux send-keys -t {num_session} \"{'exit' if not debug else 'echo Finished'}\" Enter",
            shell=True,
            universal_newlines=True
        )

    print("Done")

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", required=True, type=str)
    parser.add_argument("--benchmark_repo", type=str, default=str(Path(__file__).parent.parent))
    parser.add_argument("--write_out_base", type=str, default="./")
    parser.add_argument("--task", type=str, default="all") # one of [gsm8k, truthfulqa, triviaqa, all]
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if args.run_benchmark and not args.benchmark_repo:
        raise RuntimeError("Please specify path to repository with benchmark source (works with lm-evaluation-harness)")
    if args.endpoint_type not in ["dev", "prod", "all"]:
        raise RuntimeError("Please specify only 'dev', 'prod' or 'all' type of endpoints")

    endpoints = parse_endpoints(args.endpoints_file)

    if args.write_table:
        spreadsheet = init_gspread_client()
        today = str(date.today())
        try:
            worksheet = spreadsheet.add_worksheet(title=today, rows=250, cols=100)
        except:
            worksheet = spreadsheet.worksheet(today)
        idx = 0
        for endpoint_type in ["dev", "prod"]:
            for endpoint in endpoints[endpoint_type]:
                worksheet.update(f"A{2 + idx}", f"{endpoint_type}_{endpoint}")
                idx += 1
        worksheet.update("B1", "gsm8k")
        worksheet.update("B2:D2", ["accuracy (few shot = 0)", "accuracy (few shot = 5)", "accuracy (few shot = 8)"])
        worksheet.update("E1", "truthfulqa_gen")
        worksheet.update("E2:I2", ["bleurt_accuracy (few shot = 0)", "bleu_accuracy (few shot = 0)", "rouge1_accuracy (few shot = 0)", "rouge2_accuracy (few shot = 0)", "rougeL_accuracy (few shot = 0)"])
        worksheet.update("J1", "triviaqa")
        worksheet.update("J2:K2", ["accuracy (few shot = 0)", "accuracy (few shot = 5)"])

        
    for task in FEWSHOTS_PER_TASK.keys():
        for fewshot in FEWSHOTS_PER_TASK[task]:
            run_benchmark(
                endpoints,
                args.endpoint_type,
                args.benchmark_repo,
                num_fewshot=fewshot,
                write_out_base_path=args.write_out_base,
                task=task,
                limit=args.limit_sessions,
                write_table=args.write_table,
                debug=True
            )
    

if __name__ == "__main__":
    main()