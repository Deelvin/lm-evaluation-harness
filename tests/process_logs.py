from typing import List, Dict, NoReturn
from datetime import date
from utils import init_gspread_client
import argparse

import gspread

def get_test_names(worksheet: gspread.spreadsheet.worksheet) -> List[str]:
    return worksheet.col_values(1)

def process_test_logs(
        path_to_log: str, 
        worksheet: gspread.spreadsheet.worksheet,
        col_num: int,
        model_name: str
    ) -> None:
    test_names = get_test_names(worksheet=worksheet)
    results = [1] * len(test_names)
    with open(path_to_log, 'r') as file:
        for num_test, test_name in enumerate(test_names):
            for line in file:
                if "FAILED" in line and test_name in line:
                    results[num_test] = 0
    for i in range(len(results)):
        worksheet.format(f"{chr(ord('B') + col_num)}{2 + i}", {
            "backgroundColor": {
                "red": 0.0 if results[i] else 1.0,
                "green": 1.0 if results[i] else 0.0,
                "blue": 0.0
            }
        })

def process_benchmark_logs(
        path_to_log: str
    ) -> Dict[str, float]:
    pass

def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_log", type=str, required=True)
    parser.add_argument("--col_num", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    spreadsheet = init_gspread_client()
    today = str(date.today())
    worksheet = spreadsheet.worksheet(today)
    process_test_logs(
        path_to_log=args.path_to_log,
        worksheet=worksheet,
        col_num=args.col_num,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()