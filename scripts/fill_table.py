from typing import Dict, Any
import argparse
import datetime
import json
import os

import pandas as pd

TASKS = ["gsm8k", "truthfulqa_gen", "triviaqa"]
TASK_CONFIG = {
    "gsm8k": {
        "start_column": {0: "B", 5: "C", 8: "D"},
        "paper_results_column": "E",
        "prev_results_column": "G",
        "metrics": ["acc"],
    },
    "truthfulqa_gen": {
        "start_column": {0: "I"},
        "prev_results_column": "N",
        "metrics": ["bleurt_acc", "bleu_acc", "rouge1_acc", "rouge2_acc", "rougeL_acc"],
    },
    "triviaqa": {"start_column": {0: "P", 5: "Q"}, "prev_results_column": "R", "metrics": ["em"]},
    "human_eval": {
        "start_column": {0: "T"},
        "paper_results_column": "U",
        "prev_results_column": "W",
        "metrics": ["acc"],
    },
}


def get_row_by_model_name(worksheet: Any, model_name: str) -> int:
    models = worksheet.col_values(1)
    return models.index(model_name) + 1


def get_paper_results(spreadsheet: Any) -> Dict[str, Dict[str, float]]:
    worksheet = spreadsheet.worksheet("Paper Results")
    models = worksheet.col_values(1)
    gsm8k_results = worksheet.col_values(2)
    human_eval_results = worksheet.col_values(4)
    result_by_model = {}
    for idx, model in enumerate(models):
        if model != "" and model not in result_by_model:
            result_by_model[model] = {}
        else:
            continue
        result_by_model[model]["gsm8k"] = None
        result_by_model[model]["human_eval"] = None
        if idx < len(gsm8k_results):
            result_by_model[model]["gsm8k"] = gsm8k_results[idx]
        if idx < len(human_eval_results):
            result_by_model[model]["human_eval"] = human_eval_results[idx]
    return result_by_model


def fill_diff_from_paper(
    spreadsheet: Any,
    worksheet: Any,
    task: str,
    model_name: str,
    row: int,
) -> None:
    if task not in ["gsm8k", "human_eval"]:
        return
    result_by_model = get_paper_results(spreadsheet)
    for model in result_by_model:
        if model in model_name and result_by_model[model][task]:
            result = result_by_model[model][task]
            break
    if result is None:
        return
    paper_res_col = TASK_CONFIG[task]["paper_results_column"]
    worksheet.update(paper_res_col + str(row), result)
    diff_cell = chr(ord(paper_res_col) + 1) + str(row)
    worksheet.update(
        diff_cell, f"={TASK_CONFIG[task]['start_column'][0]}{row}-{paper_res_col}{row}", raw=False
    )
    diff_value = worksheet.acell(diff_cell).value
    statistics_worksheet = spreadsheet.worksheet(f"Statistics {task}")
    col_names = statistics_worksheet.row_values(24)  # row with column names for diffs by endpoint
    dates = statistics_worksheet.col_values(1)
    if worksheet.title not in dates:
        statistics_worksheet.update(f"A{len(dates) + 1}", worksheet.title, raw=False)
    dates = statistics_worksheet.col_values(1)
    stat_diff_cell = chr(ord("A") + col_names.index(model_name)) + str(
        dates.index(worksheet.title) + 1
    )
    statistics_worksheet.update(stat_diff_cell, diff_value, raw=False)


def fill_diff_from_prev(
    spreadsheet: Any,
    worksheet: Any,
    task: str,
    model_name: str,
    row: int,
) -> None:
    prev_res_col = TASK_CONFIG[task]["prev_results_column"]
    prev_worksheet = spreadsheet.worksheets()[1]  # 0 is current, 1 is previous
    prev_result = prev_worksheet.acell(
        TASK_CONFIG[task]["start_column"][0]
        + str(get_row_by_model_name(prev_worksheet, model_name))
    ).value
    worksheet.update(prev_res_col + str(row), prev_result)
    diff_cell = chr(ord(prev_res_col) + 1) + str(row)
    worksheet.update(
        diff_cell, f"={TASK_CONFIG[task]['start_column'][0]}{row}-{prev_res_col}{row}", raw=False
    )


def process_benchmark_results(
    path_to_results: str,
    write_out_base: str,
    model_name: str,
    write_table: bool = True,
    debug_table: bool = False,
) -> None:
    if write_table:
        from utils import init_gspread_client
        spreadsheet = init_gspread_client()
        today = str(datetime.date.today())
        table_name = "debug_table" if debug_table else today
        worksheet = spreadsheet.worksheet(table_name)
    path_to_results_root = write_out_base
    artifacts_dir = os.path.join(path_to_results_root, "results_per_task")
    with open(path_to_results, "r", encoding="utf-8") as file:
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
                current_row = get_row_by_model_name(worksheet, model_name)
                worksheet.update(
                    f"{chr(ord(start_column) + idx)}{current_row}",
                    res_file["results"][task_name][metric],
                )
                fill_diff_from_paper(
                    spreadsheet=spreadsheet,
                    worksheet=worksheet,
                    task=task_name,
                    model_name=model_name,
                    row=current_row,
                )
                fill_diff_from_prev(
                    spreadsheet=spreadsheet,
                    worksheet=worksheet,
                    task=task_name,
                    model_name=model_name,
                    row=current_row,
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
