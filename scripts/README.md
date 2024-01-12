# Running lm-evaluation-harness with OctoAI Endpoints

You can simply run *gsm8k*, *truthfulqa_gen* or *triviaqa* and get results as artifact `summary.csv` or fill corresponding Google Spreadsheet

## How to run

### Run in docker container

```bash
docker pull ailurus1/lm-eval-octoai:latest
mkdir lm_eval_results
docker run \
    --volume=./lm_eval_results:/logs \
    --env-file=.env \
    ailurus1/lm-eval-octoai:latest
```
This will run all models and tasks specified in default `config.yaml`. You can change set of models or tasks by providing your own config file as `--config` argument at the end of command.  
In `.env` file you should set environment variables `OCTOAI_TOKEN_<endpoint type>` with OctoAI tokens. All results, including the main artifact `summary.csv` will be saved in `./results` directory.  
  
__Notes:__
- Currently this method does not support filling Google Spreadsheet. Only `summary.csv` will be generated
- All arguments mentioned in the next section are also available. Just put it at the end of run command

### Run manually
0) At the first launch:  
    0.1) Create Google Cloud project and enable Google API. Configure OAuth and get credentials. [[Guide](https://developers.google.com/sheets/api/quickstart/python#enable_the_api)]  
    0.2) Place `credentials.json` in `~/.config/gspread` so `gspread` will be able to automatically detect it  
    0.3) `pip install pandas gspread gspread_dataframe gspread_formatting`  
1) `export OCTOAI_TOKEN_PROD=...`  and (or) `export OCTOAI_TOKEN_DEV=...`  
2) `python scripts/run_endpoints_benchmark.py --endpoint_type=all --write_table`  (better run from root of repo)  

__Notes:__  

- At the first launch, you will need to log in to your Google account through an automatically opening browser  
- If you want to change parameters from default config you can `--config=<path to your config.yaml>`
- By default it uses all available CPU cores to process samples in parallel session but you can specify the number of processes manually: `--limit_sessions=<number of sessions>`
- Also you can specify the number of samples to use in benchmark with `--limit_samples=<number of samples>`
- Path to logs and artifacts can be specified with `--write_out_base_path=<your path>`
- If you don't want to fill spreadsheet then just remove `--write_table`. Summary artifacts will be generated anyway.

## Logs and artifacts

By default all logs are saved in `lm-evaluation-harness/logs`. Example of generated logs:

```
├──logs
   └──summary.csv                            # main summary table
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
