from typing import NoReturn, Dict, List
from pathlib import Path
from scripts.utils import init_gspread_client
import json
import datetime
import os
import re
import pickle
import subprocess
import argparse

import gspread
import libtmux

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
        table_start_column: int = 0,
        limit: int = 3,
        debug: bool = False
    ) -> None:
    tmux_server = libtmux.Server()
    os.environ["OCTOAI_TOKEN"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}")
    current_session = 0
    for current_session, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        print()
        print(f"  ------------------------------------------------------------------------")
        print(f"| Running smoke_tests for {model_name}")
        print(f"  ------------------------------------------------------------------------")
        if current_session < limit:
            tmux_server.new_session(
                session_name=str(current_session),
                kill_session=False,
                attach=False
            )
        
        model_log_dir = os.path.join(write_out_base_path, f"{endpoint_type}_{model_name}")
        if not os.path.exists(model_log_dir):
            os.makedirs(model_log_dir)
        log_file = os.path.join(
            model_log_dir, 
            f'test_{endpoint_type}_{model_name}_{str(datetime.datetime.now()).replace(" ", "_")}.log'
        )
        print(f"Logs from this run will be saved in the following way: {log_file}")
        print()

        process_logs_command = f"""python {os.path.join(str(Path(__file__).parent.parent), 'scripts', 'process_logs.py')} \
                                   --path_to_log={log_file} \
                                   --model_name={endpoint_type}_{model_name}"""

        tmux_server.sessions[current_session % limit].panes[0].send_keys(
            f"python3 -m pytest {path_to_tests_file} " 
            f"--model_name={model_name} --endpoint={endpoint['url']} > {log_file} "
            f"&& {process_logs_command} "
            f"{'&& exit' if current_session >= len(endpoints[endpoint_type]) - limit else ''}",
            enter=True
        )
    while len(tmux_server.sessions) > 0: # wait until all tmux sessions running tests are killed
        pass

    subprocess.run(
        f"""python {os.path.join(str(Path(__file__).parent.parent), 'scripts', 'process_logs.py')} \
        --create_summary \
        --path_to_artifacts={os.path.join(write_out_base_path, 'test_results')} \
        --summary_path={os.path.join(write_out_base_path, "summary.csv")} \
        {'--write_table' if write_table else ''}""",
        shell=True
    )
    print("Done")

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", required=True, type=str)
    parser.add_argument("--tests_file", type=str, default=os.path.join(str(Path(__file__).parent.parent), 'tests_ollm/smoke_tests.py'))
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_out_base", type=str, default="./logs")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN_DEV") and not os.environ.get("OCTOAI_TOKEN_PROD"):
        raise RuntimeError("Please export your OctoAI token to environment variable OCTOAI_TOKEN_DEV or OCTOAI_TOKEN_PROD depending on type of endpoints you are going to test")

    if not os.path.exists(args.tests_file):
        raise FileNotFoundError("Specified test file not found")
    if args.endpoint_type not in ["dev", "prod", "all"]:
        raise ValueError("Please specify only 'dev', 'prod' or 'all' type of endpoints")
    
    if not os.path.exists(args.write_out_base):
        os.makedirs(args.write_out_base)

    endpoints = parse_endpoints(args.endpoints_file)

    pytest_nodes = subprocess.check_output(
        f"pytest {args.tests_file} --model_name="" --endpoint=test --collect-only",
        shell=True,
        text=True
    )
    test_names = []
    for line in pytest_nodes.split('\n'):
        match = re.search(r"test_[a-zA-Z_\-\[\]0-9\.]+", line)
        if match:
            test_names.append(match[0])
    with open(os.path.join(args.write_out_base, "test_names"), "wb") as file:
        pickle.dump(test_names, file)
    if args.write_table:
        today = str(datetime.date.today())
        spreadsheet = init_gspread_client()
        try:
            worksheet = spreadsheet.add_worksheet(title=today, rows=250, cols=100)
        except:
            worksheet = spreadsheet.worksheet(today)
        splitted_test_names = [[test_name] for test_name in test_names]
        worksheet.update(f"A2:A{2 + len(test_names)}", splitted_test_names)

    if args.endpoint_type == "all":
        run_smoke_tests(
            endpoints, 
            "dev",
            args.tests_file, 
            write_out_base_path=args.write_out_base,
            write_table=args.write_table,
            limit=args.limit_sessions,
            debug=args.debug
        )
        run_smoke_tests(
            endpoints, 
            "prod",
            args.tests_file, 
            write_out_base_path=args.write_out_base,
            write_table=args.write_table,
            table_start_column=len(endpoints["dev"]),
            limit=args.limit_sessions,
            debug=args.debug
        )
    else:
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
