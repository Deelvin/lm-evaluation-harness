from typing import List, Dict, NoReturn
from datetime import date
from utils import init_gspread_client
import argparse
import re
import time
import json

import gspread

TASKS = ["gsm8k", "truthfulqa_gen", "triviaqa"]
COLUMN_BY_TASK = {
    "gsm8k": {
        0: 'B', 
        5: 'C', 
        8: 'D'
    }, 
    "truthfulqa_gen": {
        0: 'E'
    },
    "triviaqa": {
        0: 'J',
        5: 'K'
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
        worksheet: gspread.spreadsheet.Worksheet,
        model_name: str
    ) -> None:
    with open(path_to_results) as file:
        res_file = json.load(file)
        for task in TASKS:
            try:
                task_name = res_file["results"][task]
                num_fewshot = res_file["config"]["num_fewshot"]
            except:
                continue
        worksheet.update(
            f"{COLUMN_BY_TASK[task_name][num_fewshot]}{get_row_by_model_name(worksheet, model_name)}", 
            res_file["results"][task_name]
        )

def main():
    print("Processing logs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_reults", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    spreadsheet = init_gspread_client()
    today = str(date.today())
    worksheet = spreadsheet.worksheet(today)
    process_benchmark_results(
        path_to_log=args.path_to_log,
        worksheet=worksheet,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()