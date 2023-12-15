from multimedbench.utils import Benchmark
from multimedbench.utils import Benchmark, batchSampler, Params
from tqdm import tqdm
import math
from torchmetrics import F1Score, AUROC
import torch

import os
import pandas as pd
import datasets
import json
from multimedbench.chexbert.label import label
from PIL import Image
import pydicom
import numpy as np
import urllib.request
import zipfile
import shutil


class ImageClassification(Benchmark):
    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        predictions = []
        groundTruth = []

        # Run the batcher for all data split in chunks
        for batch in tqdm(
            batchSampler(self.dataset, params.batch_size),
            total=math.ceil(len(self.dataset) / params.batch_size),
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            correctAnswers = [self.getCorrectAnswer(sample) for sample in batch]

            for idx, answer in enumerate(answers):
                predictions.append(self.getPredictedAnswer(answer))
                groundTruth.append(correctAnswers[idx])

                print(f"{predictions=}")
                print(f"{groundTruth=}")

                raise Exception

        # Convert pred and gt to tensor
        predictions = torch.tensor(predictions)
        predictions = torch.nn.functional.one_hot(predictions, num_classes=self.num_classes)
        groundTruth = torch.tensor(groundTruth)

        f1Scorer = F1Score(num_classes=self.num_classes, average="macro")
        f1Macro = f1Scorer(predictions, groundTruth)

        aurocScorer = AUROC(num_classes=self.num_classes, average="macro")
        auroc = aurocScorer(predictions, groundTruth)

        answersLog = zip(predictions, groundTruth)

        metrics = {"AUC-macro": auroc, "F1-macro": f1Macro}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def getCorrectAnswer(self, sample) -> int:
        pass

    def format_question(self, sample):
        pass

    def getPredictedAnswer(self, answer) -> int:
        """Converts the free form text output to the answer index

        Args:
            sample (_type_): The sample used to generate the answer
            answer (_type_): The free form text output of the model

        Returns:
            int: The index of the answer
        """
        pass


class MIMIC_CXR_ImageClassification(ImageClassification):
    def __init__(self, engine, **kwargs) -> None:
        super().__init__(engine, **kwargs)

        self.taskName = "MIMIC Image Classification"
        self.num_classes = 2
        self.path = json.load(open("MedMD_config.json", "r"))["MIMIC-CXR"]["path"]

        self.chexbertPath = os.path.join(
            json.load(open("MedMD_config.json", "r"))["CheXBert"]["dlLocation"], "chexbert.pth"
        )

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        chexbertMimic = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-chexpert.csv"))
        chexbertMimic = chexbertMimic[chexbertMimic.study_id.isin(split.study_id)]

        # Add the dicomid column from split in the chexbertMimic dataframe such that the studyid matches
        chexbertMimic = chexbertMimic.merge(split, on=["study_id", "subject_id"])

        self.dataset = datasets.Dataset.from_pandas(chexbertMimic)

        self.labelNames = chexbertMimic.columns.tolist()[2:]
        self.conditions = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]

    def format_question(self, sample):
        samplePath = os.path.join(
            self.path, "files", "p" + str(sample["subject_id"])[:2], "p" + str(sample["subject_id"])
        )
        imagePath = os.path.join(samplePath, "s" + str(sample["study_id"]), sample["dicom_id"] + ".jpg")
        question = "<img> List the conditions that can be seen on this picture."
        formattedText = [
            {
                "role": "user",
                "content": question,
            }
        ]

        return (formattedText, [Image.open(imagePath)])

    def getPredictedAnswer(self, answer: str) -> int:
        df = pd.DataFrame(columns=["Report Impression"], data=[answer])
        labels = [element[0] == 1 for element in label(self.chexbertPath, df)]
        labels = [labels[self.labelNames.index(condition)] for condition in self.conditions]

        if any(labels):
            return 1
        else:
            return 0

    def getCorrectAnswer(self, sample) -> int:
        # Features: [Atelectasis, cardiomegaly, consolidation, edema, and pleural effusion]
        # If any of the features is 1, then it is positive
        # If all the features are 0, -1 or NaN, then it is negative

        labels = [int(sample[condition]) == 1 for condition in self.conditions if sample[condition] is not None]

        if any(labels):
            return 1
        else:
            return 0


class VinDr_Mammo(ImageClassification):
    def __init__(self, engine, **kwargs) -> None:
        super().__init__(engine, **kwargs)

        self.taskName = "VinDr Mammo Image Classification"
        self.path = json.load(open("MedMD_config.json", "r"))["VinDr-Mammo"]["path"]

        self.num_classes = 6

        # Open the finding_annotation.csv file
        annotations = pd.read_csv(os.path.join(self.path, "finding_annotations.csv"))
        annotations = annotations[annotations["split"] == "test"]

        self.dataset = datasets.Dataset.from_pandas(annotations)

    def format_question(self, sample):
        formattedText = [
            {
                "role": "user",
                "content": f"<img>. What is the BI-RADS level from 1 to 5??",
            }
        ]

        dicom = pydicom.dcmread(os.path.join(self.path, "images", sample["study_id"], f"{sample['image_id']}.dicom"))

        dicom.BitsStored = 16
        data = dicom.pixel_array

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        image = Image.fromarray(data)
        return (formattedText, [image])

    def getPredictedAnswer(self, answer: str) -> int:
        # Find the numbers in the string answer
        findings = [int(s) for s in answer.split() if s.isdigit()]
        if len(findings) > 0:
            return findings[0]
        else:
            return 0

    def getCorrectAnswer(self, sample) -> int:
        findings = sample["finding_birads"]
        findings = int(findings[-1])

        return findings


class Pad_UFES_20(ImageClassification):
    def __init__(self, engine, **kwargs) -> None:
        super().__init__(engine, **kwargs)

        self.taskName = "Pad UFES 20 Image Classification"
        self.num_classes = 6

        self.path = json.load(open("MedMD_config.json", "r"))["Pad-UFES-20"]["path"]

        # Check if the folder contains the zip file
        if not os.path.exists(os.path.join(self.path, "pad_ufes_20.zip")):
            self._generateDataset()

        self.dataset = pd.read_csv(os.path.join(self.path, "metadata.csv"))
        print(self.dataset.columns)

        images = os.listdir(os.path.join(self.path, "images"))
        print(len(images))

        self.dataset = datasets.Dataset.from_pandas(self.dataset)

    def format_question(self, sample):
        question = f"Given <img> and the following condition of the patient:"

        options = "Options:\n" + "\n".join(
            [
                "Basal Cell Carcinoma (BCC)",
                "Squamous Cell Carcinoma (SCC)",
                "Actinic Keratosis (ACK)",
                "Seborrheic Keratosis (SEK)",
                "Bowen’s disease (BOD)",
                "Melanoma (MEL)",
                "Nevus (NEV)",
            ]
        )
        print(sample)

        image = Image.open(os.path.join(self.path, "images", sample["image_id"] + ".jpg"))

        formattedText = [{"role": "user", "content": f"{question}\n{options}"}]

        return (formattedText, [image])

    def _generateDataset(self):
        dataFolder = self.path
        # Download the file
        print("Downloading the dataset...")
        url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]) as t:
            os.makedirs(dataFolder, exist_ok=True)
            urllib.request.urlretrieve(
                url, os.path.join(dataFolder, "pad_ufes_20.zip"), reporthook=lambda x, y, z: t.update(y)
            )

        # Extract the file
        print("Extracting the dataset...")
        with zipfile.ZipFile(os.path.join(dataFolder, "pad_ufes_20.zip"), "r") as zip_ref:
            zip_ref.extractall(f"{dataFolder}")

        print("Extracting the images...")
        for file in os.listdir(os.path.join(dataFolder, "images")):
            if not file.endswith(".zip"):
                continue
            with zipfile.ZipFile(os.path.join(dataFolder, "images", file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataFolder, "images"))
                os.remove(os.path.join(dataFolder, "images", file))

        print("Copying the images...")
        for file in os.listdir(os.path.join(dataFolder, "images")):
            if not os.path.isdir(os.path.join(dataFolder, "images", file)):
                continue
            for image in os.listdir(os.path.join(dataFolder, "images", file)):
                shutil.copyfile(
                    os.path.join(dataFolder, "images", file, image), os.path.join(dataFolder, "images", image)
                )
                os.remove(os.path.join(dataFolder, "images", file, image))

            os.rmdir(os.path.join(dataFolder, "images", file))
