import argparse
import datetime
import json
import os

import gspread
import pandas as pd

from utils import init_gspread_client

TASKS = ["gsm8k", "truthfulqa_gen", "triviaqa"]
TASK_CONFIG = {
    "gsm8k": {"start_column": {0: "B", 5: "C", 8: "D"}, "metrics": ["acc"]},
    "truthfulqa_gen": {
        "start_column": {0: "I"},
        "metrics": ["bleurt_acc", "bleu_acc", "rouge1_acc", "rouge2_acc", "rougeL_acc"],
    },
    "triviaqa": {"start_column": {0: "P", 5: "Q"}, "metrics": ["em"]},
    "human_eval": {"start_column": {0: "T"}, "metrics": ["acc"]}
}


def get_row_by_model_name(
    worksheet: gspread.spreadsheet.Worksheet, model_name: str
) -> int:
    models = worksheet.col_values(1)
    return models.index(model_name) + 1


def process_benchmark_results(
    path_to_results: str,
    write_out_base: str,
    model_name: str,
    write_table: bool = True,
    debug_table: bool = False,
) -> None:
    if write_table:
        spreadsheet = init_gspread_client()
        today = str(datetime.date.today())
        table_name = "debug_table" if debug_table else today
        worksheet = spreadsheet.worksheet(table_name)
    path_to_results_root = write_out_base
    artifacts_dir = os.path.join(path_to_results_root, "results_per_task")
    with open(path_to_results, 'r', encoding="utf-8") as file:
        res_file = json.load(file)
        for task in TASKS:
            if task in res_file["results"]:
                task_name = task
                num_fewshot = res_file["config"]["num_fewshot"]
        start_column = TASK_CONFIG[task_name]["start_column"][num_fewshot]
        current_results = {}
        for idx, metric in enumerate(TASK_CONFIG[task_name]["metrics"]):
            current_results[metric] = res_file["results"][task_name][metric]
            if write_table:
                worksheet.update(
                    f"{chr(ord(start_column) + idx)}{get_row_by_model_name(worksheet, model_name)}",
                    res_file["results"][task_name][metric],
                )
        current_results["endpoint"] = model_name
        results_dataframe = pd.DataFrame(current_results, index=[0])
        artifact_path = os.path.join(artifacts_dir, f"{task_name}_summary.csv")
        if os.path.exists(artifact_path):
            temp_dataframe = pd.read_csv(artifact_path)
            results_dataframe = pd.concat(
                [temp_dataframe, results_dataframe], axis=0, ignore_index=True
            )
        if not os.path.exists(os.path.dirname(artifact_path)):
            os.makedirs(os.path.dirname(artifact_path))
        results_dataframe.to_csv(artifact_path, index=False)


def main():
    print("Processing results...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, required=True)
    parser.add_argument("--write_out_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--debug_table", action="store_true")
    args = parser.parse_args()

    process_benchmark_results(
        path_to_results=args.path_to_results,
        write_out_base=args.write_out_base,
        model_name=args.model_name,
        write_table=args.write_table,
        debug_table=args.debug_table,
    )


if __name__ == "__main__":
    main()
