import datasets
from . import triviaqa

import os


class TruncatedTriviaQA(triviaqa.TriviaQA):
    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Go up two directory levels
    parent_dir = os.path.join(script_directory, os.pardir, os.pardir)

    # Define the path relative to the script location
    relative_path = "tests/testdata/triviaqa_truncated_mistral.json"
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
            "json",
            data_files=self.DATASET_PATH,
            data_dir=data_dir,
            cache_dir=cache_dir,
            split="train",
        )

        print(self.dataset)
