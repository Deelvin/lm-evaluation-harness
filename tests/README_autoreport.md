# Running all tests for OctoAI endpoints with autoreporting

## How to run

Before run make sure that you have all needed golden files in `tests_ollm` and `endpoints.json` with the following structure:
```
"dev": [
    {
        "url": <url for prod>,
        "model": <endpoint name>,
        "context_size": <maximum supported size of context>
    },
    ...
],
"prod": [
    {
        "url": <url for prod>,
        "model": <endpoint name>,
        "context_size": <maximum supported size of context>
    },
    ...
]
```
You can simply run all tests and get results as artifact `logs/summary.csv` or fill corresponding Google Spreadsheet  
0) At the first launch:  
    0.1) Create Google Cloud project and enable Google API. Configure OAuth and get credentials. [[Guide](https://developers.google.com/sheets/api/quickstart/python#enable_the_api)]  
    0.2) Place `credentials.json` in `~/.config/gspread` so `gspread` will be able to automatically detect it  
    0.3) `pip install libtmux pandas gspread gspread_dataframe gspread_formatting`
1) `export PYTHONPATH=<path to lm-evaluation-harness local repo>:$PYTHONPATH`  
2) `export OCTOAI_TOKEN_PROD=...`  and (or) `export OCTOAI_TOKEN_DEV=...`  
3) `python tests/test_endpoints.py --write_table --error_notes`  (better run from root of repo)  

__Notes:__

- At the first launch, you will need to log in to your Google account through an automatically opening browser  
- you can specify `--endpoint_type` is one of `['prod', 'dev', 'all']`. `all` is default  
- If your file with endpoints not in this directory then you can specify it with `--endpoints_file=<path to your endpoints.json>`
- Script uses tmux sessions to run different endpoints in parallel. By default it uses 4 sessions but you can specify this number manually: `--limit_sessions=<number of sessions>`
- Path to logs and artifacts can be specified with `--write_out_base=<your path>`
- If you don't want to fill spreadsheet then just remove `--write_table`. Summary artifact will be generated anyway  
- If you want to report errors for failed tests in spredsheet then you can use flag `--error_notes`  
- For developing new tests or when you just don't want to rewrite current results than you can use flag `--debug` to redirect results to new worksheet called `debug_table`  

## Logs and artifacts

By default all logs are saved in `lm-evaluation-harness/logs`. Example of generated logs:

```
├──logs
   └──dev_codellama-7b-instruct-fp16
      ├──test_dev_codellama-7b-instruct-fp16_2023-12-08_00:00:00.00.log
      └──test_dev_codellama-7b-instruct-fp16_2023-12-08_00:00:01.00.log
    ...
   └──<endpoint_type>_<endpoint_name>
      ├──test_<endpoint_type>_<endpoint_name>_2023-12-08_00:00:00.00.log
      └──test_<endpoint_type>_<endpoint_name>_2023-12-08_00:00:01.00.log
    ...
    └──test_results
       ├──results_dev_codellama-7b-instruct-fp16.csv
       ...
       ├──results_<endpoint_type>_<endpoint_name>.csv
       └──errors
           └──dev_codellama-7b-instruct-fp16_error_messages
             └──test_valid_temperature_error
             └──...
           ...
           └──<endpoint_type>_<endpoint_name>_error_messages
              └──<test_case>_error
       └──summary.csv                                       # Main summary artifact
       └──test_names

```