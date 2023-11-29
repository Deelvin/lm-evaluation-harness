from typing import List, Dict
from datetime import date
from pathlib import Path
import argparse
import re
import os
import pickle

import gspread_dataframe as gd
import gspread_formatting as gf
import pandas as pd

from utils import init_gspread_client

def get_test_names(path_to_log_dir: str) -> List[str]:
    with open(os.path.join(path_to_log_dir, "test_names"), "rb") as file:
        test_names = pickle.load(file)
    return test_names

def extract_error_messages(path_to_log: str) -> Dict[str, str]:
    error_messages = {}
    with open(path_to_log, 'r') as file:
        current_message = ""
        current_test_name = ""
        for line in file:
            if line.startswith("___") and line.endswith("___"):
                current_test_name = line.replace("_", "").replace(" ", "")
                current_message += line
            if line.startswith("___") and line.endswith("___") or \
               line.startswith("===") and line.endswith("==="):
                error_messages[current_test_name] = current_message
                current_test_name, current_message = "", ""
    return error_messages

def process_test_logs(
        path_to_log: str,
        model_name: str,
    ) -> None:
    path_to_log_root = str(Path(path_to_log).parent.parent)
    test_names = get_test_names(path_to_log_dir=path_to_log_root)
    results = ["Passed"] * len(test_names)
    error_messages = extract_error_messages(path_to_log=path_to_log)
    for num_test, test_name in enumerate(test_names):
        if test_name in error_messages.keys():
            results[num_test] = "Failed"
    artifacts_dir = os.path.join(path_to_log_root, "test_results")
    errors_dir = os.path.join(artifacts_dir, f"{model_name}_error_messages")
    if not os.path.exists(errors_dir):
        os.makedirs(errors_dir)
    pd.DataFrame(
        {
            "test_case": test_names,
            model_name: results
        }
    ).set_index("test_case").to_csv(os.path.join(artifacts_dir, f"results_{model_name}.csv"))
    

def create_summary(
        path_to_artifacts: str,
        path_to_summary: str,
        remove_artifacts: bool = True
    ) -> None:
    print("Creating summary...")
    results = []
    for filename in sorted(os.listdir(path_to_artifacts)):
        if filename.startswith("results_") and filename.endswith(".csv"):
            results.append(pd.read_csv(os.path.join(path_to_artifacts, filename)))
    summary = results[0].set_index("test_case")
    for i in range(1, len(results)):
        summary = summary.join(
            results[i].set_index("test_case"),
            on="test_case"
        )
    summary.to_csv(path_to_summary)
    if remove_artifacts:
        for filename in os.listdir(path_to_artifacts):
            if filename.startswith("results_") and filename.endswith(".csv"):
                os.remove(os.path.join(path_to_artifacts, filename))
    print()
    print(f"  ------------------------------------------------------------------------")
    print(f"| Summary is stored in {os.path.abspath(path_to_summary)}")
    print(f"  ------------------------------------------------------------------------")
    print()

def write_table(
        path_to_summary: str,
        path_to_errors: str = None,
        debug_table: bool = False,
    ) -> None:
    print("Writing table...")
    spreadsheet = init_gspread_client()
    today = str(date.today())
    table_name = "debug_table" if debug_table else today
    worksheet = spreadsheet.worksheet(table_name)
    rules = gf.get_conditional_format_rules(worksheet)
    rules.clear()
    rule_passed = gf.ConditionalFormatRule(
        ranges=[gf.GridRange.from_a1_range('A1:CT100', worksheet)],
        booleanRule=gf.BooleanRule(
            condition=gf.BooleanCondition("TEXT_CONTAINS", ["Passed"]),
            format=gf.CellFormat(backgroundColor=gf.Color(0,1,0))
        )
    )
    rule_failed = gf.ConditionalFormatRule(
        ranges=[gf.GridRange.from_a1_range('A1:CT100', worksheet)],
        booleanRule=gf.BooleanRule(
            condition=gf.BooleanCondition("TEXT_CONTAINS", ["Failed"]),
            format=gf.CellFormat(backgroundColor=gf.Color(1,0,0))
        )
    )
    rules.append(rule_passed), rules.append(rule_failed)
    rules.save()
    results = pd.read_csv(path_to_summary)
    gd.set_with_dataframe(worksheet, results)
    # TODO: add search for cell coordinates by test and model names and put a note with error message there
    print("Done")

def main() -> None:
    print("Processing logs...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_log", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--create_summary", action="store_true")
    parser.add_argument("--path_to_artifacts", type=str)
    parser.add_argument("--remove_artifacts", action="store_true")
    parser.add_argument("--summary_path", type=str)
    parser.add_argument("--write_table", action="store_true")
    args = parser.parse_args()

    if args.create_summary:
        create_summary(
            path_to_artifacts=args.path_to_artifacts,
            path_to_summary=args.summary_path,
            remove_artifacts=args.remove_artifacts
        )

    if args.write_table:
        write_table(args.summary_path)

    if not args.write_table and not args.create_summary:
        process_test_logs(
            path_to_log=args.path_to_log,
            model_name=args.model_name,
        )

if __name__ == "__main__":
    main()
