"""The MedNLI task family."""

import os
import zipfile

import pandas as pd
from datasets import Dataset

from multimedeval.task_families import QA
from multimedeval.utils import download_file


class MedNLI(QA):
    """The MedNLI task family."""

    def __init__(self, **kwargs):
        """Initializes the MedNLI task family."""
        super().__init__(**kwargs)
        self.task_name = "MedNLI"
        self.modality = "General medicine"
        self.task = "NLI"

    def setup(self):
        """Setups the MedNLI task family."""
        self.path = self.engine.get_config()["mednli_dir"]

        if self.path is None:
            raise ValueError(
                "No path for MedNLI dataset provided in the config file. Skipping the task."
            )

        self._generate_dataset()

        test_set = pd.read_json(
            path_or_buf=os.path.join(self.path, "mli_test_v1.jsonl"), lines=True
        )

        self.options = test_set["gold_label"].unique().tolist()

        test_set = test_set[["sentence1", "sentence2", "gold_label"]]
        self.dataset = Dataset.from_pandas(test_set)

        train_set = pd.read_json(
            path_or_buf=os.path.join(self.path, "mli_train_v1.jsonl"), lines=True
        )
        train_set = train_set[["sentence1", "sentence2", "gold_label"]]
        self.train_dataset = Dataset.from_pandas(train_set)

    def format_question(self, sample, prompt=False):
        """Formats the question for the MedNLI task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer to the prompt. Defaults to False.

        Returns:
            The formatted question and images.
        """
        formatted_question = (
            f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\n"
        )
        formatted_question += (
            "Determine the logical relationship between these two sentences. "
            "Does the second sentence logically follow from the first (entailment), "
            "contradicts the first (contradiction), or if there is no clear logical "
            "relationship between them (neutral)?"
        )

        question = [{"role": "user", "content": formatted_question}]
        if prompt:
            formatted_answer = (
                "The logical relationship is " + sample["gold_label"] + "."
            )
            question.append({"role": "assistant", "content": formatted_answer})

        return (question, [])

    def get_correct_answer(self, sample, full_text=False):
        """Returns the correct answer for the MedNLI task.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Return the raw answer. Defaults to False.

        Returns:
            The correct answer.
        """
        return sample["gold_label"]

    def get_predicted_answer(self, pred: str, sample):
        """Returns the predicted answer for the MedNLI task.

        Args:
            pred: The prediction to get the answer for.
            sample: The sample corresponding to the prediction.

        Returns:
            The predicted answer.
        """
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
