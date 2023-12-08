from typing import List, Dict, Tuple
from datetime import date
from pathlib import Path
import argparse
import re
import os
import pickle

import gspread
import gspread_dataframe as gd
import gspread_formatting as gf
import pandas as pd

from autoreport.utils import init_gspread_client


def get_test_names(path_to_log_dir: str) -> List[str]:
    with open(os.path.join(path_to_log_dir, "test_names"), "rb") as file:
        test_names = pickle.load(file)
    return test_names


def remove_token(message: str) -> str:
    """
    Remove token from pytest error messages for security purposes
    """
    print("DEBUG")
    print(message)
    token_line = re.search(r"token\s+=\s+'[A-Za-z0-9\.\-\,\_]+'", message)[0]
    print("\n\n" + token_line + "\n\n")
    token_stub = "token = <token was removed from this message during log processing>"
    return message.replace(token_line, token_stub)


def get_test_case_cell(
    worksheet: gspread.spreadsheet.Worksheet, model_name: str, test_case: str
) -> Tuple[int, int]:
    """
    Takse row with endpoint names (columns' names)
    and column with test case names (rows' names)
    to find coordinates (indexes) of a cell by given parameters
    """
    endpoint_columns = worksheet.row_values(1)
    test_rows = worksheet.col_values(1)
    # Make sure parameters really exist in table
    assert model_name in endpoint_columns
    assert test_case in test_rows
    return test_rows.index(test_case) + 1, endpoint_columns.index(model_name) + 1


def prepare_error_notes(
    path_to_errors: str, worksheet: gspread.spreadsheet.Worksheet
) -> Dict[str, str]:
    """
    Go over files in given error directory
    and get following Dict:
        {<row and column cooridnates in A1 notation>: error message}
    to insert into table
    """
    model_name = path_to_errors.split("/")[-1][: -len("_error_messages")]
    error_notes = {}
    for filename in os.listdir(path_to_errors):
        test_case = filename[: -len("_error")]
        with open(os.path.join(path_to_errors, filename), 'r', encoding="utf-8") as file:
            row, col = get_test_case_cell(worksheet, model_name, test_case)
            s = file.read()
            error_notes[
                gspread.utils.rowcol_to_a1(row=row, col=col)
            ] = s
    return error_notes


def extract_error_messages(path_to_log: str) -> Dict[str, str]:
    error_messages = {}
    with open(path_to_log, 'r', encoding="utf-8") as file:
        current_message = ""
        current_test_name = ""
        inside_error = False
        for line in file:
            if line.startswith("___") and line.endswith("___\n") and not inside_error:
                current_test_name = line.strip("_ \n")
                inside_error = True
                current_message += line
                continue
            if inside_error and (
                line.startswith("___")
                and line.endswith("___\n")
                or line.startswith("===")
                and line.endswith("===\n")
            ):
                error_messages[current_test_name] = remove_token(current_message)
                inside_error = False
                current_test_name, current_message = "", ""
            if inside_error:
                current_message += line
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
    errors_dir = os.path.join(artifacts_dir, "errors", f"{model_name}_error_messages")
    if not os.path.exists(errors_dir):
        os.makedirs(errors_dir)
    for test_case in error_messages:
        with open(os.path.join(errors_dir, f"{test_case}_error"), 'w', encoding="utf-8") as file:
            file.write(error_messages[test_case])
    pd.DataFrame({"test_case": test_names, model_name: results}).set_index("test_case").to_csv(
        os.path.join(artifacts_dir, f"results_{model_name}.csv")
    )


def create_summary(
    path_to_artifacts: str, path_to_summary: str, remove_artifacts: bool = True
) -> None:
    print("Creating summary...")
    results = []
    for filename in sorted(os.listdir(path_to_artifacts)):
        if filename.startswith("results_") and filename.endswith(".csv"):
            results.append(pd.read_csv(os.path.join(path_to_artifacts, filename)))
    summary = results[0].set_index("test_case")
    for i in range(1, len(results)):
        summary = summary.join(results[i].set_index("test_case"), on="test_case")
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
        ranges=[gf.GridRange.from_a1_range("A1:CT100", worksheet)],
        booleanRule=gf.BooleanRule(
            condition=gf.BooleanCondition("TEXT_CONTAINS", ["Passed"]),
            format=gf.CellFormat(backgroundColor=gf.Color(0, 1, 0)),
        ),
    )
    rule_failed = gf.ConditionalFormatRule(
        ranges=[gf.GridRange.from_a1_range("A1:CT100", worksheet)],
        booleanRule=gf.BooleanRule(
            condition=gf.BooleanCondition("TEXT_CONTAINS", ["Failed"]),
            format=gf.CellFormat(backgroundColor=gf.Color(1, 0, 0)),
        ),
    )
    rules.append(rule_passed), rules.append(rule_failed)
    rules.save()
    results = pd.read_csv(path_to_summary)
    gd.set_with_dataframe(worksheet, results)
    # Inserting notes with error messages into table
    if path_to_errors is not None:
        error_notes = {}
        for directory in os.listdir(path_to_errors):
            error_notes |= prepare_error_notes(os.path.join(path_to_errors, directory), worksheet)
        if len(error_notes) > 1:
            worksheet.update_notes(error_notes)
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
    parser.add_argument("--debug_table", action="store_true")
    parser.add_argument("--error_notes", action="store_true")
    args = parser.parse_args()

    if args.create_summary:
        create_summary(
            path_to_artifacts=args.path_to_artifacts,
            path_to_summary=args.summary_path,
            remove_artifacts=args.remove_artifacts,
        )

    if args.write_table:
        if args.error_notes:
            path_to_errors = os.path.join(args.path_to_artifacts, "errors")
        else:
            path_to_errors = None
        write_table(
            path_to_summary=args.summary_path,
            path_to_errors=path_to_errors,
            debug_table=args.debug_table,
        )

    if not args.write_table and not args.create_summary:
        process_test_logs(
            path_to_log=args.path_to_log,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
