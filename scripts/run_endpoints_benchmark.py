from typing import Dict, List
from pathlib import Path
import json
import os
import datetime
import subprocess
import argparse

import libtmux

from utils import init_gspread_client

FEWSHOTS_PER_TASK = {
    "gsm8k": [0, 5, 8],
    "truthfulqa_gen": [0],
    "triviaqa": [0, 5],
}


def parse_endpoints(
    path_to_endpoints_file: str,
) -> Dict[str, List[Dict[str, str]]]:
    with open(path_to_endpoints_file, "r+", encoding="utf-8") as file:
        endpoints = json.load(file)
    return endpoints


def run_benchmark(
    endpoints: Dict[str, List[Dict[str, str]]],
    endpoint_type: str,
    path_to_benchmark_repo: str,
    num_fewshot: int = 0,
    write_out_base_path: str = "./logs",
    task: str = "gsm8k",
    write_table: bool = True,
    debug: bool = False,
    limit: int = 3,
) -> None:
    tmux_server = libtmux.Server()
    os.environ["OCTOAI_API_KEY"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}", "")
    assert os.environ["OCTOAI_API_KEY"] != "", "OctoAI token is not specified"
    for num_endpoint, endpoint in enumerate(endpoints[endpoint_type]):
        model_name: str = endpoint["model"]
        res_path = os.path.join(write_out_base_path, task, model_name)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        work_dir = os.getcwd()
        os.chdir(res_path)

        if not os.path.exists(f"./nf{num_fewshot}"):
            os.makedirs(f"./nf{num_fewshot}")

        print()
        print("  -------------------------------------------------------------------------------")
        print(f"| Running benchmark for {endpoint_type}_{model_name}, num_fewshot={num_fewshot}")
        print(f"| Results from this run will be saved in the following path: {res_path}")
        print("  -------------------------------------------------------------------------------")
        print()

        if num_endpoint < limit:
            subprocess.run(
                f"tmux new-session -d -s {num_endpoint} ",
                shell=True,
                universal_newlines=True,
                check=False
            )

        write_table_command = ""

        res_output = os.path.join(
            f"nf{num_fewshot}",
            f"{endpoint_type}_{model_name}_{str(datetime.datetime.now()).replace(' ', '_')}.json",
        )

        process_logs_script = os.path.join(str(Path(__file__).parent), 'process_logs.py')
        write_out_abs = os.path.join(work_dir, write_out_base_path)
        if write_table:
            write_table_command = f"""  python {process_logs_script} \
                                        --path_to_results={res_output} \
                                        --model_name={endpoint_type}_{model_name} \
                                        {'--write_table' if write_table else ''} \
                                        {'--debug_table' if debug else ''} \
                                        --write_out_base={write_out_abs}"""

        extra_args = "--limit=0.1" if task == "triviaqa" else ""

        tmux_server.sessions[num_endpoint % limit].panes[0].send_keys(
            f"python3 {path_to_benchmark_repo}/main.py "
            f"--model=octoai "
            f"--model_args='model_name={model_name},prod={str(endpoint_type == 'prod')}' "
            f"--task={task} "
            f"--output_path={res_output} "
            f"--no_cache "
            f"--num_fewshot={num_fewshot} "
            f"--batch_size=1 "
            f"--write_out "
            f"{extra_args} "
            f"--output_base_path={Path(res_output).parent}/",
            enter=True,
        )

        tmux_server.sessions[num_endpoint % limit].panes[0].send_keys(
            write_table_command, enter=True
        )

        os.chdir(work_dir)

    while len(tmux_server.sessions) > 0:  # wait until all tmux sessions running tests are killed
        for session in tmux_server.sessions:
            if session.panes[0].capture_pane()[-1].endswith("$"):
                session.panes[0].send_keys("exit", enter=True)

    print("Done")


def main() -> None: # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints_file", type=str, default=os.path.join(str(Path(__file__).parent), "endpoints.json"))
    parser.add_argument("--benchmark_repo", type=str, default=str(Path(__file__).parent.parent))
    parser.add_argument("--write_out_base", type=str, default="./logs")
    parser.add_argument("--task", type=str, default="all")  # [gsm8k, truthfulqa, triviaqa, all]
    parser.add_argument("--endpoint_type", type=str, default="dev")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN_DEV") and not os.environ.get("OCTOAI_TOKEN_PROD"):
        raise RuntimeError(
            """Please export your OctoAI token to environment variable 
            OCTOAI_TOKEN_DEV or OCTOAI_TOKEN_PROD depending on type of 
            endpoints you are going to test"""
        )

    if args.endpoint_type not in ["dev", "prod", "all"]:
        raise RuntimeError("Please specify only 'dev', 'prod' or 'all' type of endpoints")

    endpoints: Dict[str, List[Dict[str, str]]] = parse_endpoints(args.endpoints_file)
    chosen_types = ["dev", "prod"] if args.endpoint_type == "all" else [args.endpoint_type]

    if args.write_table:
        spreadsheet = init_gspread_client()
        today = str(datetime.date.today())
        table_name = "debug_table" if args.debug else today
        try:
            worksheet = spreadsheet.worksheet(table_name)
        except: # pylint: disable=bare-except
            worksheet = spreadsheet.worksheet("Template").duplicate(new_sheet_name=table_name)
        idx = 0
        for endpoint_type in chosen_types:
            for endpoint in endpoints[endpoint_type]:
                worksheet.update(f"A{3 + idx}", f"{endpoint_type}_{endpoint['model']}")
                idx += 1

    for num_fewshot in [0, 5, 8]:
        for endpoint_type in chosen_types:
            for task in FEWSHOTS_PER_TASK:
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
                        debug=args.debug,
                    )


if __name__ == "__main__":
    main()
