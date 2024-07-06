"""This module contains the dynamic datasets for the MultiMedEval framework."""

import json
import os

from PIL import Image
from torchmetrics.text import BLEUScore

from multimedeval.task_families import QA, VQA, ImageClassification, ReportComparison
from multimedeval.utils import clean_str


def make_constructor(new_class, dataset, dataset_name, dataset_samples):
    """Create the constructor for the dynamic class."""

    def constructor(self, **kwargs) -> None:
        super(new_class, self).__init__(**kwargs)
        self.path = dataset
        self.taskName = dataset_name
        self.modality = dataset_samples["modality"]
        self.dataset = dataset_samples["samples"]

    return constructor


def setup(self):  # noqa # pylint: disable=unused-argument
    """Setup the dataset."""


def format_question(self, sample, prompt=False):
    """Formats the question for the user and the assistant.

    Args:
        sample: The sample to format.
        prompt: Add the answer to the prompt. Defaults to False.

    Returns:
        A tuple with the formatted prompt and the images.
    """
    if "images" in sample:
        images = [
            Image.open(os.path.join(self.path, imagePath))
            for imagePath in sample["images"]
        ]
    else:
        images = []

    question = " ".join(["<img>" for _ in images]) + " " + sample["question"]

    if "options" in sample:
        question += " " + " ".join(sample["options"])

    formatted_prompt = [{"role": "user", "content": question}]

    if prompt:
        formatted_prompt.append({"role": "assistant", "content": sample["answer"]})

    return (formatted_prompt, images)


def get_correct_answer(
    self, sample, full_text=False
):  # pylint: disable=unused-argument
    """Returns the correct answer for the sample."""
    return sample["answer"].lower().strip()


def get_predicted_answer(self, pred: str, sample):
    """Converts the free form text output to the answer index."""
    pred = clean_str(pred)
    if len(pred) == 0:
        return "Invalid answer"

    options = [clean_str(option) for option in sample["options"]]
    # Compute the BLEU score for each option
    scores = [self.bleuScorer([pred], [[option]]) for option in options]

    if max(scores) == 0:
        return "Invalid answer"

    pred = sample["options"][scores.index(max(scores))].lower()

    return pred


def find_datasets():
    """Find the additional datasets in the MultiMedEvalAdditionalDatasets folder."""
    # Find the subfolders in the MultiMedEvalAdditionalDatasets folder
    additional_datasets_path = "MultiMedEvalAdditionalDatasets"

    if not os.path.exists(additional_datasets_path):
        return []

    additional_datasets = [
        f.path for f in os.scandir(additional_datasets_path) if f.is_dir()
    ]

    print("Found the following additional datasets:" + str(additional_datasets))

    # For each datasets, create a dynamic class with the dataset name
    dynamic_datasets = []
    for dataset in additional_datasets:
        dataset_name = dataset.split("/")[-1]

        # Open the dataset.json file
        with open(dataset + "/dataset.json", "r", encoding="utf-8") as json_file:
            dataset_samples = json.load(json_file)

        task_family = dataset_samples["taskType"]

        attributes = {
            "setup": setup,
            "format_question": format_question,
            "get_correct_answer": get_correct_answer,
        }

        if task_family == "VQA":
            parentClass = VQA
        elif task_family == "QA":
            parentClass = QA
            attributes["get_predicted_answer"] = get_predicted_answer
            attributes["bleu_scorer"] = BLEUScore(n_gram=1)
        elif task_family == "ImageClassification":
            raise NotImplementedError(
                "ImageClassification task family not implemented yet"
            )
            parentClass = ImageClassification  # pylint: disable=unreachable
        elif task_family == "ReportComparison":
            raise NotImplementedError(
                "ReportComparison task family not implemented yet"
            )
            parentClass = ReportComparison  # pylint: disable=unreachable
        else:
            raise ValueError("Task family not recognized")

        new_class = type(dataset_name, (parentClass,), attributes)

        new_class.__init__ = make_constructor(
            new_class, dataset, dataset_name, dataset_samples
        )
        dynamic_datasets.append(new_class)

    return dynamic_datasets
