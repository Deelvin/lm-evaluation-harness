import datasets
from . import gsm8k
from pathlib import Path

import os

class TruncatedGradeSchoolMath8K(gsm8k.GradeSchoolMath8K):
    # Go up two directory levels
    parent_dir = Path(__file__).parents[2]

    # Define the path relative to the script location
    relative_path = "tests/testdata/gsm8k_truncated_llama.json"
    DATASET_PATH = os.path.join(parent_dir, relative_path)


    def has_training_docs(self):
        return False

    def has_test_docs(self):
        return True    

    def training_docs(self):
        return NotImplementedError

    def test_docs(self):
        return self.dataset

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        self.dataset = datasets.load_dataset(
            "json",data_files=self.DATASET_PATH,
            data_dir=data_dir,
            cache_dir=cache_dir,
            split="train"
        )

        print(self.dataset)