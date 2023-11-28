from typing import List, Dict, NoReturn
from datetime import date
from utils import init_gspread_client
from pathlib import Path
import argparse
import re
import time
import json
import os

import gspread
import pandas as pd

TASKS = ["gsm8k", "truthfulqa_gen", "triviaqa"]
TASK_CONFIG = {
    "gsm8k": {
        "column_by_fewshot": {
            0: 'B', 
            5: 'C', 
            8: 'D'
        },
        "metrics": ["acc"]
    },
    "truthfulqa_gen": {
        "column_by_fewshot": {
            0: 'G'
        },
        "metrics": [
            "bleurt_acc", 
            "bleu_acc", 
            "rouge1_acc", 
            "rouge2_acc", 
            "rougeL_acc"
        ]
    },
    "triviaqa": {
        "column_by_fewshot": {
            0: 'N',
            5: 'O'
        },
        "metrics": ["em"]
    }
}

def get_row_by_model_name(
        worksheet: gspread.spreadsheet.Worksheet, 
        model_name: str
    ) -> int:
    models = worksheet.col_values(1)
    return models.index(model_name) + 1

def process_benchmark_results(
        path_to_results: str, 
        model_name: str,
        write_table: bool = True,
        debug_table: bool = False
    ) -> None:
    if write_table:
        spreadsheet = init_gspread_client()
        today = str(date.today())
        table_name = "debug_table" if debug_table else today
        worksheet = spreadsheet.worksheet(table_name)
    path_to_results_root = str(Path(path_to_results).parent.parent.parent)
    artifacts_dir = os.path.join(path_to_results_root, "results_per_task")
    with open(path_to_results) as file:
        res_file = json.load(file)
        for task in TASKS:
            try:
                temp_val = res_file["results"][task]
                task_name = task
                num_fewshot = res_file["config"]["num_fewshot"]
            except:
                continue
        start_column = TASK_CONFIG[task_name]["column_by_fewshot"][num_fewshot]
        current_results = dict()
        for idx, metric in enumerate(TASK_CONFIG[task_name]["metrics"]):
            current_results[metric] = res_file["results"][task_name][metric]
            if write_table:
                worksheet.update(
                    f"{chr(ord(start_column) + idx)}{get_row_by_model_name(worksheet, model_name)}", 
                    res_file["results"][task_name][metric]
                )
        current_results["endpoint"] = model_name
        results_dataframe = pd.DataFrame(current_results, index=[model_name])
        artifact_path = os.path.join(artifacts_dir, f"{task_name}_summary.csv")
        if os.path.exists(artifact_path):
            temp_dataframe = pd.read_csv(artifact_path)
            results_dataframe = pd.concat([temp_dataframe, results_dataframe], axis=1, ignore_index=False)
        results_dataframe.to_csv(artifact_path)

def main():
    print("Processing results...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--write_table", action="store_true")
    parser.add_argument("--debug_table", action="store_true")
    args = parser.parse_args()

    process_benchmark_results(
        path_to_results=args.path_to_results,
        model_name=args.model_name,
        write_table=args.write_table,
        debug_table=args.debug_table
    )

if __name__ == "__main__":
    main()