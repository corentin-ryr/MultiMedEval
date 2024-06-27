import json
import os

from PIL import Image
from torchmetrics.text import BLEUScore

from multimedeval.taskFamilies import QA, VQA, ImageClassification, ReportComparison
from multimedeval.utils import Benchmark, cleanStr


def make_constructor(newClass, dataset, datasetName, datasetSamples):
    def constructor(self, **kwargs) -> None:
        super(newClass, self).__init__(**kwargs)
        self.path = dataset
        self.taskName = datasetName
        self.modality = datasetSamples["modality"]
        self.dataset = datasetSamples["samples"]

    return constructor


def setup(self):
    pass


def format_question(self, sample, prompt=False):

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

    formattedPrompt = [{"role": "user", "content": question}]

    if prompt:
        formattedPrompt.append({"role": "assistant", "content": sample["answer"]})

    return (formattedPrompt, images)


def getCorrectAnswer(self, sample, fullText=False):
    return sample["answer"].lower().strip()


def getPredictedAnswer(self, pred: str, sample):
    pred = cleanStr(pred)
    if len(pred) == 0:
        return "Invalid answer"

    options = [cleanStr(option) for option in sample["options"]]
    # Compute the BLEU score for each option
    scores = [self.bleuScorer([pred], [[option]]) for option in options]

    if max(scores) == 0:
        return "Invalid answer"

    pred = sample["options"][scores.index(max(scores))].lower()

    return pred


def findDatasets():
    # Find the subfolders in the MultiMedEvalAdditionalDatasets folder
    additionalDatasetsPath = "MultiMedEvalAdditionalDatasets"

    if not os.path.exists(additionalDatasetsPath):
        return []

    additionalDatasets = [
        f.path for f in os.scandir(additionalDatasetsPath) if f.is_dir()
    ]

    print("Found the following additional datasets:" + str(additionalDatasets))

    # For each datasets, create a dynamic class with the dataset name
    dynamicDatasets = []
    for dataset in additionalDatasets:
        datasetName = dataset.split("/")[-1]

        # Open the dataset.json file
        datasetSamples = json.load(open(dataset + "/dataset.json"))
        taskFamily = datasetSamples["taskType"]

        attributes = {
            "setup": setup,
            "format_question": format_question,
            "getCorrectAnswer": getCorrectAnswer,
        }

        if taskFamily == "VQA":
            parentClass = VQA
        elif taskFamily == "QA":
            parentClass = QA
            attributes["getPredictedAnswer"] = getPredictedAnswer
            attributes["bleuScorer"] = BLEUScore(n_gram=1)
        elif taskFamily == "ImageClassification":
            raise NotImplementedError(
                "ImageClassification task family not implemented yet"
            )
            parentClass = ImageClassification
        elif taskFamily == "ReportComparison":
            raise NotImplementedError(
                "ReportComparison task family not implemented yet"
            )
            parentClass = ReportComparison
        else:
            raise ValueError("Task family not recognized")

        newClass = type(datasetName, (parentClass,), attributes)

        newClass.__init__ = make_constructor(
            newClass, dataset, datasetName, datasetSamples
        )
        dynamicDatasets.append(newClass)

    return dynamicDatasets
