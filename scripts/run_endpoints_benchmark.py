from typing import Dict, List, Optional, Union, TypeAlias
from pathlib import Path
import os
import datetime
import subprocess
import argparse

import yaml

FEWSHOTS_PER_TASK = {
    "gsm8k": [0, 5, 8],
    # "human_eval": [0],
    "triviaqa": [0, 5],
    "truthfulqa_gen": [0],
}

GSM8K_SIZE = 1319
TRUTHFULQA_GEN_SIZE = 817
TRIVIAQA_SIZE = 17944
HUMANEVAL_SIZE = 163

DEFAULT_CONFIG = str(Path(__file__).parent.joinpath("config.yaml"))

Config: TypeAlias = Dict[str, Union[Dict[str, List[str]], List[str]]]


def process_config(config: Config) -> Config:
    """
    Validate the structure of config and fill
    missing parameters with default values
    """
    assert list(config.keys()) in [
        ["models", "tasks"],
        ["models"],
        ["tasks"],
    ], 'Config can contain only "models" and/or "tasks" fields'
    if "models" in config:
        assert list(config["models"]) in [
            ["dev", "prod"],
            ["dev"],
            ["prod"],
        ], 'Need to specify type of endpoint environment ("dev" or "prod")'
    with open(DEFAULT_CONFIG, "r+", encoding="utf-8") as file:
        default_config = yaml.safe_load(file)
    default_config.update(config)
    return default_config


def parse_config(
    path_to_config_file: str,
) -> Config:
    with open(path_to_config_file, "r+", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def run_benchmark(
    models: Dict[str, List[str]],
    endpoint_type: str,
    path_to_benchmark_repo: str,
    num_fewshot: int = 0,
    write_out_base_path: str = str(Path(__file__).resolve().parents[1].joinpath("logs")),
    task: str = "gsm8k",
    write_table: bool = True,
    debug: bool = False,
    limit_sessions: int = 2,
    limit_samples: Optional[int] = None,
) -> None:
    os.environ["OCTOAI_TOKEN"] = os.environ.get(f"OCTOAI_TOKEN_{endpoint_type.upper()}", "")
    assert os.environ["OCTOAI_TOKEN"] != "", "OctoAI token is not specified"
    cmds: List[str] = []
    for num_endpoint, endpoint in enumerate(models[endpoint_type]):
        res_path = Path(write_out_base_path) / task / f"{endpoint_type}_{endpoint}"
        if not res_path.exists():
            res_path.mkdir(parents=True)
        work_dir = Path.cwd()

        nf_path = Path(res_path / f"nf{num_fewshot}")
        if not nf_path.exists():
            nf_path.mkdir(parents=True)

        print()
        print("  -------------------------------------------------------------------------------")
        print(f"| Running benchmark for {endpoint_type}_{endpoint}, num_fewshot={num_fewshot}")
        print(f"| Results from this run will be saved in the following path: {res_path}")
        print("  -------------------------------------------------------------------------------")
        print()

        cmds.append("")
        cmds[num_endpoint] += f"cd {res_path};"

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

        cmds[
            num_endpoint
        ] += f""" python {path_to_benchmark_repo}/main.py \
            --model=octoai \
            --model_args='model_name={endpoint},prod={str(endpoint_type == 'prod')},batch_size={limit_sessions}' \
            --task={task} \
            --output_path={res_output} \
            --no_cache \
            --num_fewshot={num_fewshot} \
            --write_out \
            {extra_args} \
            --output_base_path={Path(res_output).parent}/; """
        cmds[num_endpoint] += write_table_command
        cmds[num_endpoint] += f" cd {work_dir}; "
    for cmd in cmds:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            check=True
        )
    print("\nDone")


def main() -> None:  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
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
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--limit_sessions", type=int, default=os.cpu_count())
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OCTOAI_TOKEN_DEV") and not os.environ.get("OCTOAI_TOKEN_PROD"):
        raise RuntimeError(
            """Please export your OctoAI token to environment variable 
            OCTOAI_TOKEN_DEV or OCTOAI_TOKEN_PROD depending on type of 
            endpoint environment you are going to test"""
        )
    config: Config = process_config(parse_config(args.config))
    chosen_types = config["models"].keys()

    if args.write_table:
        from utils import init_gspread_client # pylint: disable=import-outside-toplevel

        spreadsheet = init_gspread_client()
        today = str(datetime.date.today())
        table_name = "debug_table" if args.debug else today
        try:
            worksheet = spreadsheet.worksheet(table_name)
        except:  # pylint: disable=bare-except
            worksheet = spreadsheet.worksheet("Template").duplicate(new_sheet_name=table_name)
        idx = 0
        for endpoint_type in chosen_types:
            for model in config["models"][endpoint_type]:
                worksheet.update(f"A{3 + idx}", f"{endpoint_type}_{model}")
                idx += 1

    for num_fewshot in [0, 5, 8]:
        for endpoint_type in chosen_types:
            for task in config["tasks"]:
                if num_fewshot in FEWSHOTS_PER_TASK[task]:
                    run_benchmark(
                        config["models"],
                        endpoint_type,
                        args.benchmark_repo,
                        num_fewshot=num_fewshot,
                        write_out_base_path=args.write_out_base,
                        task=task,
                        # bleu metrics take a lot of CPU resources so manually set only 1 process to run
                        limit_sessions=args.limit_sessions if task != "truthfulqa_gen" else 1,
                        write_table=args.write_table,
                        debug=args.debug,
                        limit_samples=args.limit_samples,
                    )


if __name__ == "__main__":
    main()
