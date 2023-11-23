from typing import NoReturn, Dict, List
from pathlib import Path
from utils import init_gspread_client
import json
import datetime
import os
import re
import subprocess
import argparse

import gspread

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
        limit: int = 3,
        debug: bool = False
    ) -> None:
    if not os.environ.get("OCTOAI_TOKEN"):
        os.environ["OCTOAI_TOKEN"] = os.environ.get("OCTOI_API_KEY")
    current_session = 0
    for col_num, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        print(f"--------------------------------------------------------------------------")
        print(f"Running smoke_tests for {model_name}")
        print(f"--------------------------------------------------------------------------")
        if current_session < limit:
            subprocess.run(
                f"tmux new-session -d -s {current_session} ",
                shell=True, 
                universal_newlines=True
            )
        
        model_log_dir = os.path.join(write_out_base_path, model_name)
        if not os.path.exists(model_log_dir):
            os.makedirs(model_log_dir)
        log_file = os.path.join(
            model_log_dir, 
            f'test_{model_name}_{str(datetime.now()).replace(" ", "_")}.log'
        )
        print(f"Logs from this run will be saved in the following way: {log_file}")

        write_table_command = ""
        if write_table:
            write_table_command = f"python {os.path.join(str(Path(__file__).parent), 'process_logs.py')} --path_to_log={log_file} --col_num={col_num} --model_name={model_name}"

        subprocess.run(
            f"tmux send-keys -t {current_session % limit} "
            f"\"ENDPOINT={endpoint['url']} "
            f"python3 -m pytest {path_to_tests_file} " 
            f"--model_name={model_name} > {log_file}\" Enter",
            shell=True,
            universal_newlines=True
        )

        subprocess.run(
            f"tmux send-keys -t {current_session % limit} "
            f"\"{write_table_command}\" Enter", 
            shell=True, 
            universal_newlines=True
        )

        current_session += 1
    
    for num_session in range(limit):
        subprocess.run(
            f"tmux send-keys -t {num_session} \"echo Finished\" {'C-m' if debug else ''}"
        )
    print("Done")

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", required=True, type=str)
    parser.add_argument("--tests_file", type=str, default=os.path.join(str(Path(__file__).parent.parent), 'tests/smoke_tests.py'))
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_out_base", type=str, default="./logs")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN") and not os.environ.get("OCTOAI_API_KEY"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN or to OCTOAI_API_KEY")

    if not os.path.exists(args.tests_file):
        raise FileNotFoundError("Specified test file not found")
    if args.endpoint_type not in ["dev", "prod"]:
        raise ValueError("Please specify only dev or prod type of endpoints")
    
    if not os.path.exists(args.write_out_base):
        os.mkdirs(args.write_out_base)

    endpoints = parse_endpoints(args.endpoints_file)

    if args.write_table:
        spreadsheet = init_gspread_client()
        pytest_nodes = subprocess.check_output(
            f"ENDPOINT=test pytest {args.tests_file} --model_name="" --collect-only",
            shell=True,
            text=True
        )
        test_names = []
        for line in pytest_nodes.split('\n'):
            match = re.search(r"test_[a-zA-Z_]+\[.*\]", line)
            if match:
                test_names.append(match[0])
        today = str(datetime.date.today())
        try:
            worksheet = spreadsheet.add_worksheet(title=today, rows=250, cols=100)
        except:
            worksheet = spreadsheet.worksheet(today)
        splitted_test_names = [[test_name] for test_name in test_names]
        worksheet.update(f"A2:A{2 + len(test_names)}", splitted_test_names)

    run_smoke_tests(
        endpoints, 
        args.endpoint_type,
        args.tests_file, 
        write_out_base_path=args.write_out_base,
        write_table=args.write_table,
        limit=args.limit_sessions,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
