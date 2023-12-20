# Running lm-evaluation-harness with OctoAI Endpoints

You can simply run *gsm8k*, *truthfulqa_gen* or *triviaqa* and get results as artifacts `<task_name>_summary.csv` or fill corresponding Google Spreadsheet

## How to run

0) At the first launch:  
    0.1) Create Google Cloud project and enable Google API. Configure OAuth and get credentials. [[Guide](https://developers.google.com/sheets/api/quickstart/python#enable_the_api)]  
    0.2) Place `credentials.json` in `~/.config/gspread` so `gspread` will be able to automatically detect it  
    0.3) `pip install libtmux pandas gspread gspread_dataframe gspread_formatting`  
1) `export OCTOAI_TOKEN_PROD=...`  and (or) `export OCTOAI_TOKEN_DEV=...`  
2) `python scripts/run_endpoints_benchmark.py --endpoint_type=all --write_table`  (better run from root of repo)  

__Notes:__  

- At the first launch, you will need to log in to your Google account through an automatically opening browser  
- `--endpoint_type` is one of `['prod', 'dev', 'all']`  
- use `--task=<task name>` to run particular task  
- Order of evaluation is following: first we go through increasing number of fewshots (0, 5, 8) and then for each of them we go over tasks  
- If your file with endpoints not in this directory then you can specify it with `--endpoints_file=<path to your endpoints.json>`
- Script uses tmux sessions to run different endpoints in parallel. By default it uses 4 sessions but you can specify this number manually: `--limit_sessions=<number of sessions>`
- Also you can specify the number of samples to use in benchmark with `--limit_samples=<number of samples>`
- Path to logs and artifacts can be specified with `--write_out_base_path=<your path>`
- If you don't want to fill spreadsheet then just remove `--write_table`. Summary artifacts will be generated anyway.

## Logs and artifacts

By default all logs are saved in `lm-evaluation-harness/logs`. Example of generated logs:

```
├──logs
   └──results_per_task                       # directory with summary files
      └──gsm8k_summary.csv
   └──gsm8k                                  # task name
      ├──dev_codellama-7b-instruct-fp16      # endpoint
      └──nf0                                 # number of fewshot
          ├──dev_codellama-7b-instruct-fp16_2023-11-29_16:00:00.0000.json
          └──gsm8k_write_out_info.json
      ├──dev_codellama-13b-instruct-fp16
      └──nf0
          ├──dev_codellama-7b-instruct-fp16_2023-11-29_16:00:00.0000.json
          └──gsm8k_write_out_info.json
```
