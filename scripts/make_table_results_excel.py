import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd

PARENT_DIR = str(Path(__file__).resolve().parents[1])

def parse_args():
    parser = argparse.ArgumentParser("Parse results from harness")
    parser.add_argument(
        "-out",
        "--output_dir",
        type=str,
        default="logs",
        help="Dir for output files",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to input file",
    )
    return parser.parse_args()


def make_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    task_name, task_metric_1 , task_stderr_1, task_metric_2, task_stderr_2 = [], [], [], [], []
    all_arr = [task_metric_1, task_stderr_1, task_metric_2, task_stderr_2]
    for task, res in results.items():
        task_name.append(task)
        if len(res) < 4:
            for val, arr in zip(res.values(), [task_metric_1, task_stderr_1], strict=False):
                arr.append(val)
            task_metric_2.append(0)
            task_stderr_2.append(0)
        else:
            for val, arr in zip(res.values(), all_arr, strict=False):
                arr.append(val)

    df = pd.DataFrame(
        {
        "Task": task_name,
        "metric_1": task_metric_1,
        "metic_2": task_metric_2,
        "stderr_1": task_stderr_1,
        "stderr_2": task_stderr_2,
        }
    )
    return df


def save_table(df: pd.DataFrame, output_dir: str, file_name: str) -> None:
    if not os.path.exists(f"{PARENT_DIR}/{output_dir}"):
        os.makedirs(f"{PARENT_DIR}/{output_dir}")

    df.to_excel(f"{PARENT_DIR}/{output_dir}/{file_name}.xlsx")


def main():
    args = parse_args()
    if not args.input_file and not args.input_dir:
        raise ValueError("You should set --input_file or --input_dir")
    elif args.input_file and args.input_dir:
        raise ValueError("You cann't set --input_file and --input_dir")

    if args.input_dir:
        for dirpath, dirnames, filenames in os.walk(f"{PARENT_DIR}/{args.input_dir}"):
            if not filenames:
                continue

            for filename in sorted([f for f in filenames if f.endswith(".json")]):
                path = os.path.join(dirpath, filename)
                with open(path, "r") as f:
                    result_dict = json.load(f)["results"]
                result_df = make_table(result_dict)
                save_table(result_df, args.output_dir, filename.replace(".json", ""))
    else:
        with open(args.input_file) as f:
            result_dict = json.load(f)["results"]
        result_df = make_table(result_dict)
        filename = args.input_file.split("/")[-1]
        save_table(result_df, args.output_dir, filename.replace(".json", ""))


if __name__ == "__main__":
    main()
