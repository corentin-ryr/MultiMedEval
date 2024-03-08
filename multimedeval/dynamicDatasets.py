from multimedeval.utils import Benchmark
from multimedeval.taskFamilies import VQA, QA, ImageClassification, ReportComparison
import os
import json
from PIL import Image


def setup(self):
    pass


def format_question(self, sample, prompt=False):

    formattedPrompt = [{"role": "user", "content": sample["question"]}]

    if prompt:
        formattedPrompt.append({"role": "assistant", "content": sample["answer"]})

    images = [Image.open(os.path.join(self.path, imagePath)) for imagePath in sample["images"]]

    return (formattedPrompt, images)


def getCorrectAnswer(self, sample):
    return sample["answer"].lower().strip()


def findDatasets():
    # Find the subfolders in the MultiMedEvalAdditionalDatasets folder
    additionalDatasetsPath = "MultiMedEvalAdditionalDatasets"
    additionalDatasets = [f.path for f in os.scandir(additionalDatasetsPath) if f.is_dir()]

    print("Found the following additional datasets:" + str(additionalDatasets))

    # For each datasets, create a dynamic class with the dataset name
    dynamicDatasets = set()
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
            raise NotImplementedError("QA task family not implemented yet")
            parentClass = QA
        elif taskFamily == "ImageClassification":
            raise NotImplementedError("ImageClassification task family not implemented yet")
            parentClass = ImageClassification
        elif taskFamily == "ReportComparison":
            raise NotImplementedError("ReportComparison task family not implemented yet")
            parentClass = ReportComparison
        else:
            raise ValueError("Task family not recognized")

        newClass = type(datasetName, (parentClass,), attributes)

        def constructor(self, **kwargs) -> None:
            super(newClass, self).__init__(**kwargs)
            self.path = dataset
            self.taskName = datasetName
            self.modality = datasetSamples["modality"]
            self.dataset = datasetSamples["samples"]

        newClass.__init__ = constructor
        dynamicDatasets.add(newClass)

    return dynamicDatasets
