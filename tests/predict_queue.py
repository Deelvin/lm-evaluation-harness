import subprocess

import tests.testdata.config as cfg

model_type = "octoai"
batch_size = 1
limit = 1
fewshots = [0, 5, 8]


for model_name in cfg.MODEL_NAMES:
    for task in cfg.TASKS:
        for num_fewshot in fewshots:
            print(["python3", "/home/dbarinov/lm-evaluation-harness/main.py", f"--model={model_type}", f"--model_args=model_name={model_name}", f"--tasks={task}", f"--output_path=./{model_name}-{task}-{num_fewshot}fs.json", "--no_cache", f"--num_fewshot={num_fewshot}", f"--batch_size={batch_size}", f"--limit={limit}", "--write_out"])
            subprocess.call(["python3", "/home/dbarinov/lm-evaluation-harness/main.py", f"--model={model_type}", f"--model_args=model_name={model_name}", f"--tasks={task}", f"--output_path=./{model_name}-{task}-{num_fewshot}fs.json", "--no_cache", f"--num_fewshot={num_fewshot}", f"--batch_size={batch_size}", f"--limit={limit}", "--write_out"])