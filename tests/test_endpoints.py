from typing import NoReturn, Dict
import json
import datetime
import os
import re
import pickle
import subprocess
import argparse

import libtmux

from autoreport.utils import init_gspread_client


def parse_endpoints(
    path_to_endpoints_file: str,
) -> Dict[str, str]:
    with open(path_to_endpoints_file, "r+") as file:
        endpoints = json.load(file)
    return endpoints


def _tmux_active(server: libtmux.Server) -> bool:
    for session in server.sessions:
        if session.panes[0].capture_pane()[-1].endswith("$"):
            continue
        else:
            return True
    return False


def run_smoke_tests(
    endpoints: Dict[str, str],
    endpoint_type: str,
    paths_to_test_files: List[str],
    write_out_base_path: str = "./",
    write_table: bool = True,
    limit: int = 3,
    debug: bool = False,
    error_notes: bool = False,
) -> None:
    tmux_server = libtmux.Server()
    os.environ["OCTOAI_TOKEN"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}")
    for current_session, endpoint in enumerate(endpoints[endpoint_type]):
        model_name = endpoint["model"]
        print()
        print(f"  ------------------------------------------------------------------------")
        print(f"| Running smoke_tests for {endpoint_type}_{model_name}")
        print(f"  ------------------------------------------------------------------------")
        if current_session < limit:
            tmux_server.new_session(
                session_name=str(current_session), kill_session=False, attach=False
            )

        model_log_dir = os.path.join(write_out_base_path, f"{endpoint_type}_{model_name}")
        if not os.path.exists(model_log_dir):
            os.makedirs(model_log_dir)
        log_file = os.path.join(
            model_log_dir,
            f'test_{endpoint_type}_{model_name}_{str(datetime.datetime.now()).replace(" ", "_")}.log',
        )
        print("Logs from this run will be saved in the following path:")
        print(log_file)
        print()

        process_logs_command = f"""python {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'autoreport', 'process_logs.py')} \
                                   --path_to_log={log_file} \
                                   --model_name={endpoint_type}_{model_name}"""

        run_cmd = "python3 -m pytest"
        for file in paths_to_test_files:
            run_cmd += f" {file} --model_name={model_name} --endpoint={endpoint['url']} --context_size={endpoint['context_size']}"
        tmux_server.sessions[current_session % limit].panes[0].send_keys(
            f"{run_cmd} > {log_file}",
            enter=True,
        )
        tmux_server.sessions[current_session % limit].panes[0].send_keys(
            process_logs_command, enter=True
        )
    while _tmux_active(tmux_server):  # wait until all tmux sessions running tests are killed
        continue

    while len(tmux_server.sessions) > 0:
        for session in tmux_server.sessions:
            try:
                session.panes[0].send_keys("exit", enter=True)
            except:
                continue

    subprocess.run(
        f"""python {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'autoreport', 'process_logs.py')} \
        --create_summary \
        --path_to_artifacts={os.path.join(write_out_base_path, 'test_results')} \
        --summary_path={os.path.join(write_out_base_path, "summary.csv")} \
        {'--write_table' if write_table else ''} \
        {'--debug_table' if debug else ''} \
        {'--error_notes' if error_notes else ''}""",
        shell=True,
    )
    print("Done")


def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoints_file",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tests_ollm", "endpoints.json"
        ),
    )
    parser.add_argument(
        "--test_files",
        type=str,
        nargs='*',
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tests_ollm/smoke_tests.py"
        ),
    )
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_out_base", type=str, default="./logs")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error_notes", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN_DEV") and not os.environ.get("OCTOAI_TOKEN_PROD"):
        raise RuntimeError(
            "Please export your OctoAI token to environment variable OCTOAI_TOKEN_DEV or OCTOAI_TOKEN_PROD depending on type of endpoints you are going to test"
        )

    for file in args.test_files:
        if not os.path.exists(file):
            raise FileNotFoundError("Specified test file not found")
    if args.endpoint_type not in ["dev", "prod", "all"]:
        raise ValueError("Please specify only 'dev', 'prod' or 'all' type of endpoints")

    if not os.path.exists(args.write_out_base):
        os.makedirs(args.write_out_base)

    endpoints = parse_endpoints(args.endpoints_file)

    pytest_dummy_cmd = "pytest"
    for file in args.test_files:
        pytest_dummy_cmd += f" {file} --model_name=" " --endpoint=test --collect-only"
    pytest_nodes = subprocess.check_output(
        pytest_dummy_cmd,
        shell=True,
        text=True,
    )
    test_names = []
    for line in pytest_nodes.split("\n"):
        match = re.search(r"test_[a-zA-Z_\-\[\]0-9\.!?,]+", line)
        if match:
            test_names.append(match[0])
    with open(os.path.join(args.write_out_base, "test_names"), "wb") as file:
        pickle.dump(test_names, file)
    if args.write_table:
        today = str(datetime.date.today())
        spreadsheet = init_gspread_client()
        table_name = "debug_table" if args.debug else today
        try:
            worksheet = spreadsheet.add_worksheet(title=table_name, rows=250, cols=100)
        except:
            worksheet = spreadsheet.worksheet(table_name)
        splitted_test_names = [[test_name] for test_name in test_names]
        worksheet.update(f"A2:A{2 + len(test_names)}", splitted_test_names)

    chosen_types = ["dev", "prod"] if args.endpoint_type == "all" else [args.endpoint_type]
    for endpoint_type in chosen_types:
        run_smoke_tests(
            endpoints,
            endpoint_type,
            args.test_files,
            write_out_base_path=args.write_out_base,
            write_table=args.write_table,
            limit=args.limit_sessions,
            debug=args.debug,
            error_notes=True if args.error_notes else False,
        )


if __name__ == "__main__":
    main()
