from typing import Dict, List, Optional
from pathlib import Path
import json
import os
import datetime
import subprocess
import argparse

FEWSHOTS_PER_TASK = {
    "gsm8k": [0, 5, 8],
    "truthfulqa_gen": [0],
    "triviaqa": [0, 5],
    # "human_eval": [0]
}

GSM8K_SIZE = 1319
TRUTHFULQA_GEN_SIZE = 817
TRIVIAQA_SIZE = 17944
HUMANEVAL_SIZE = 163


def parse_endpoints(
    path_to_endpoints_file: str,
) -> Dict[str, List[str]]:
    with open(path_to_endpoints_file, "r+", encoding="utf-8") as file:
        endpoints = json.load(file)
    return endpoints


def run_benchmark(
    endpoints: Dict[str, List[str]],
    endpoint_type: str,
    path_to_benchmark_repo: str,
    num_fewshot: int = 0,
    write_out_base_path: str = str(Path(__file__).resolve().parents[1].joinpath("logs")),
    task: str = "gsm8k",
    write_table: bool = True,
    debug: bool = False,
    limit_sessions: int = 3,
    limit_samples: Optional[int] = None,
) -> None:
    os.environ["OCTOAI_TOKEN"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}", "")
    assert os.environ["OCTOAI_TOKEN"] != "", "OctoAI token is not specified"
    cmds: List[str] = []
    for num_endpoint, endpoint in enumerate(endpoints[endpoint_type]):
        res_path = Path(write_out_base_path) / task / f"{endpoint_type}_{endpoint}"
        if not res_path.exists():
            res_path.mkdir(parents=True)
        work_dir = Path.cwd()
        os.chdir(res_path)

        nf_path = Path(f"./nf{num_fewshot}")
        if not nf_path.exists():
            nf_path.mkdir(parents=True)

        print()
        print("  -------------------------------------------------------------------------------")
        print(f"| Running benchmark for {endpoint_type}_{endpoint}, num_fewshot={num_fewshot}")
        print(f"| Results from this run will be saved in the following path: {res_path}")
        print("  -------------------------------------------------------------------------------")
        print()

        if num_endpoint < limit_sessions:
            cmds.append("")
        # else:
        #     cmds[num_endpoint % limit_sessions] += "; "

        res_output = str(
            Path(f"nf{num_fewshot}")
            / f"{endpoint_type}_{endpoint}_{str(datetime.datetime.now()).replace(' ', '_')}.json"
        )

        fill_table_script = str(Path(__file__).parent.joinpath("fill_table.py"))
        write_out_abs = str(Path(work_dir) / write_out_base_path)
        write_table_command = f"""  python {fill_table_script} \
                                    --path_to_results={res_output} \
                                    --model_name={endpoint_type}_{endpoint} \
                                    {'--write_table' if write_table else ''} \
                                    {'--debug_table' if debug else ''} \
                                    --write_out_base={write_out_abs}; """
            
        extra_args = f"--limit={limit_samples}" if limit_samples else ""

        cmds[num_endpoint % limit_sessions] += f""" python {path_to_benchmark_repo}/main.py \
            --model=octoai \
            --model_args='model_name={endpoint},prod={str(endpoint_type == 'prod')},batch_size={os.cpu_count() // limit_sessions}' \
            --task={task} \
            --output_path={res_output} \
            --no_cache \
            --num_fewshot={num_fewshot} \
            --write_out \
            {extra_args} \
            --output_base_path={Path(res_output).parent}/; """
        cmds[num_endpoint % limit_sessions] += write_table_command
        cmds[num_endpoint % limit_sessions] += f" cd {work_dir}; "
    # print(cmds)
    # return
    running_procs = [
        subprocess.Popen(
            cmd, 
            # stdout=subprocess.PIPE, 
            # stderr=subprocess.PIPE,
            shell=True
        ) for cmd in cmds
    ]
    while running_procs:
        for proc in running_procs:
            if proc.poll() is not None:
                running_procs.remove(proc)
                break
        print(f"\r{len(running_procs)} sessions left out of {limit_sessions} for {task}", end="")
    print("\nDone")


def main() -> None:  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoints_file",
        type=str,
        default=str(Path(__file__).parent.joinpath("endpoints.json")),
    )
    parser.add_argument(
        "--benchmark_repo", type=str, default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument(
        "--write_out_base",
        type=str,
        default=str(Path(__file__).resolve().parents[1].joinpath("logs")),
    )
    parser.add_argument(
        "--task", type=str, default="all"
    )  # [gsm8k, truthfulqa_gen, triviaqa, human_eval, all]
    parser.add_argument("--endpoint_type", type=str, default="prod")
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--limit_samples", type=int, default=None)
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
        from utils import init_gspread_client
        spreadsheet = init_gspread_client()
        today = str(datetime.date.today())
        table_name = "debug_table" if args.debug else today
        try:
            worksheet = spreadsheet.worksheet(table_name)
        except:  # pylint: disable=bare-except
            worksheet = spreadsheet.worksheet("Template").duplicate(new_sheet_name=table_name)
        idx = 0
        for endpoint_type in chosen_types:
            for endpoint in endpoints[endpoint_type]:
                worksheet.update(f"A{3 + idx}", f"{endpoint_type}_{endpoint}")
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
                        # bleu metrics take a lot of CPU resources so manually set only 1 process to generate
                        limit_sessions=args.limit_sessions if task != "truthfulqa" else 1,
                        write_table=args.write_table,
                        debug=args.debug,
                        limit_samples=args.limit_samples,
                    )


if __name__ == "__main__":
    main()
