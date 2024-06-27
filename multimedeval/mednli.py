import os
import zipfile

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from multimedeval.taskFamilies import QA
from multimedeval.tqdm_loggable import tqdm_logging
from multimedeval.utils import Benchmark, EvalParams, download_file


class MedNLI(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedNLI"
        self.modality = "General medicine"
        self.task = "NLI"

    def setup(self):
        self.path = self.engine.getConfig()["MedNLI_dir"]

        if self.path is None:
            raise Exception(
                "No path for MedNLI dataset provided in the config file. Skipping the task."
            )

        self._generate_dataset()

        testSet = pd.read_json(
            path_or_buf=os.path.join(self.path, "mli_test_v1.jsonl"), lines=True
        )

        self.options = testSet["gold_label"].unique().tolist()

        testSet = testSet[["sentence1", "sentence2", "gold_label"]]
        self.dataset = Dataset.from_pandas(testSet)

        trainSet = pd.read_json(
            path_or_buf=os.path.join(self.path, "mli_train_v1.jsonl"), lines=True
        )
        trainSet = trainSet[["sentence1", "sentence2", "gold_label"]]
        self.trainDataset = Dataset.from_pandas(trainSet)

    def format_question(self, sample, prompt=False):
        formattedQuestion = (
            f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\n"
        )
        formattedQuestion += "Determine the logical relationship between these two sentences. Does the second sentence logically follow from the first (entailment), contradicts the first (contradiction), or if there is no clear logical relationship between them (neutral)?"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            formattedAnswer = (
                "The logical relationship is " + sample["gold_label"] + "."
            )
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        return sample["gold_label"]

    def getPredictedAnswer(self, pred: str, sample):
        # Check if the prediction contains each of the options
        scores = []
        for option in self.options:
            scores.append(option in pred)

        if sum(scores) != 1:
            return "Invalid answer"

        return self.options[scores.index(True)]

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(
                self.path,
                "mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0",
                "mli_test_v1.jsonl",
            )
        ):
            self.path = os.path.join(
                self.path,
                "mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0",
            )
            return

        os.makedirs(self.path, exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        # wget_command = f'wget -d -c --user "{username}" --password "{password}" -O "{self.path}/mednli.zip"  https://physionet.org/content/mednli/get-zip/1.0.0/'
        # subprocess.run(wget_command, shell=True, check=True)

        download_file(
            "https://physionet.org/content/mednli/get-zip/1.0.0/",
            os.path.join(self.path, "mednli.zip"),
            username,
            password,
        )

        # Unzip the file
        with zipfile.ZipFile(os.path.join(self.path, "mednli.zip"), "r") as zip_ref:
            zip_ref.extractall(self.path)

        # Remove the zip file
        os.remove(os.path.join(self.path, "mednli.zip"))

        self.path = os.path.join(
            self.path,
            "mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0",
        )
