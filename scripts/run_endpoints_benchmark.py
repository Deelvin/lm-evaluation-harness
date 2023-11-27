from typing import NoReturn, Dict, List
from datetime import date
from pathlib import Path
from utils import init_gspread_client
import json
import os
import subprocess
import argparse

import gspread
import libtnux

FEWSHOTS_PER_TASK = {
    "gsm8k": [0, 5, 8], 
    "truthfulqa_gen": [0], 
    "triviaqa": [0, 5]
}

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
    tmux_server = libtmux.Server()
    os.environ["OCTOAI_API_KEY"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}")
    for num_endpoint, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        res_path = os.path.join(write_out_base_path, model_name)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        work_dir = os.getcwd()
        os.chdir(res_path)

        if not os.path.exists(os.path.join(res_path, f"nf{num_fewshot}")):
            os.makedirs(os.path.join(res_path, f"nf{num_fewshot}"))
        
        print()
        print(f"  ---------------------------------------------------------------------------------")
        print(f"| Running benchmark for {endpoint_type}_{model_name} with --num_fewshot={num_fewshot}")
        print(f"| Results from this run will be saved in the following path: {res_path}")
        print(f"  ---------------------------------------------------------------------------------")
        print()

        if num_endpoint < limit:
            subprocess.run(
                f"tmux new-session -d -s {num_endpoint} ",
                shell=True, 
                universal_newlines=True
            )

        write_table_command = ""

        res_output = os.path.join(
            res_path, 
            f"nf{num_fewshot}",
            f"{task}_nf{num_fewshot}_{endpoint_type}_{model_name}_{str(date.today()).replace(' ', '_')}.json"
        )

        if write_table:
            write_table_command = f"python {os.path.join(str(Path(__file__).parent), 'process_logs.py')} --path_to_results={res_output} --model_name={endpoint_type}_{model_name}"

        # extra_args = "--limit=0.1" if task == "triviaqa" else ""
        extra_args = "--limit=8"

        tmux_server.sessions[num_endpoint % limit].panes[0].send_keys(
            f"python3 {path_to_benchmark_repo}/main.py "
            f"--model=octoai "
            f"--model_args=\'model_name={model_name},prod={True if endpoint_type == 'prod' else False}\' "
            f"--task={task} "
            f"--output_path={res_output} " 
            f"--no_cache " 
            f"--num_fewshot={num_fewshot} " 
            f"--batch_size=1 " 
            f"--write_out " 
            f"{extra_args} "
            f"--output_base_path=./nf{num_fewshot}/", 
            enter=True
        )

        tmux_server.sessions[num_endpoint % limit].panes[0].send_keys(
            write_table_command,
            enter=True
        )
        
        os.chdir(work_dir)

    while len(tmux_server.sessions) > 0: # wait until all tmux sessions running tests are killed
        for session in tmux_server.sessions:
            if session.panes[0].capture_pane()[-1].endswith("$"):
                session.panes[0].send_keys("exit", enter=True)

    print("Done")

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", required=True, type=str)
    parser.add_argument("--benchmark_repo", type=str, default=str(Path(__file__).parent.parent))
    parser.add_argument("--write_out_base", type=str, default="./")
    parser.add_argument("--task", type=str, default="all") # one of [gsm8k, truthfulqa, triviaqa, all]
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if args.endpoint_type not in ["dev", "prod", "all"]:
        raise RuntimeError("Please specify only 'dev', 'prod' or 'all' type of endpoints")

    endpoints = parse_endpoints(args.endpoints_file)

    if args.write_table:
        spreadsheet = init_gspread_client()
        today = str(date.today())
        try:
            worksheet = spreadsheet.worksheet(today)
        except:
            worksheet = spreadsheet.worksheet("Template").duplicate(new_sheet_name=today)
        idx = 0
        for endpoint_type in ["dev", "prod"]:
            for endpoint in endpoints[endpoint_type]:
                worksheet.update(f"A{3 + idx}", f"{endpoint_type}_{endpoint['model']}")
                idx += 1

    chosen_types = ["dev", "prod"] if args.endpoint_type == "all" else [args.endpoint_type]
    for num_fewshot in [0, 5, 8]:
        for endpoint_type in chosen_types:
            for task in FEWSHOTS_PER_TASK.keys():
                if num_fewshot in FEWSHOTS_PER_TASK[task]:
                    run_benchmark(
                        endpoints,
                        endpoint_type,
                        args.benchmark_repo,
                        num_fewshot=num_fewshot,
                        write_out_base_path=args.write_out_base,
                        task=task,
                        limit=args.limit_sessions,
                        write_table=args.write_table,
                        debug=True
                    )
    

if __name__ == "__main__":
    main()