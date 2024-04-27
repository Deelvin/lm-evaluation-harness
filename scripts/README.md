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

## Running loglikelihood benchmarks
### Run on h100
- commands for mlc-serve:
```bash
/opt/bin/cuda-reserve.py --num-gpus <N> python3 -m mlc_serve --artifact-name <model_dir> --port 9001
```
`N` - the number of `--tensor-parallel-shards` from model build

```bash
python main.py \ 
--model octoai \
--model_args model_name=<model_name>,url=http://127.0.0.1:9001,batch_size=16 \
--num_fewshot=0 \
--tasks <list_of_tasks> \
--no_cache \
--write_out \
--output_base_path <path_for_answers> \
--output_path <path_for_result>
```

- commands for hugginface backend:
```bash
/opt/bin/cuda-reserve.py --num-gpus <N> python3 -m main \
--model hf-causal-experimental \
--device auto \
--model_args pretrained=/opt/models/<model> \
--batch_size=16 \
--num_fewshot=0 \
--tasks <list_of_tasks> \
--no_cache \
--write_out \
--output_base_path <path_for_answers> \
--output_path <path_for_result>
```

However, at the moment, there are .bin files available for llama-70b. In order to use the model,  you will need to add `use_safetensors=False` [here](https://github.com/Deelvin/lm-evaluation-harness/blob/bafc4d61280904824b92a0387f0e4e0ff87705fb/lm_eval/models/huggingface.py#L283).

The list of all tasks (bigbench, mmlu, hellaswag, arc, winogrande, truthfulqa):

```
bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_dyck_languages,bigbench_formal_fallacies_syllogisms_negation,bigbench_hyperbaton,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions,truthfulqa_mc,hellaswag,winogrande,arc_easy,arc_challenge
```

### To parse the results
To parse the results into an Excel table, use the script `make_table_results_excel.py`. You can either parse a single file (`--input_file`) or a folder containing the results (`--input_dir`). Please note that the final files will be named after the original .json files. 

The default dir for saving is `.logs/`. You can change it with flag `--output_dir`.