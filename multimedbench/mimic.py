import logging
import os
from multimedbench.utils import Benchmark
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import datasets

from multimedbench.utils import Benchmark, batchSampler, Params, remove_punctuation
import math
from torchmetrics.text import BLEUScore, ROUGEScore

import csv


class MIMIC_CXR_reportgen(Benchmark):
    def __init__(self, seed=1111):
        logging.debug("***** Transfer task : MIMIC_CXR *****\n\n")
        self.seed = seed

        self.taskName = "MIMIC_CXR report generation"

        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys="rougeL")

        self.f1 = []

        # Get the dataset ====================================================================
        self.path = json.load(open("MedMD_config.json", "r"))["MIMIC-CXR"]["path"]

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        findings = {}
        with open(os.path.join(self.path, "mimic_findings.csv"), "r") as f:
            spamreader = csv.reader(f, delimiter=",")
            for line in spamreader:
                findings[line[0]] = line[1]


        # For each sample in the dataset, open its report and check if it contains the keyword "findings"
        self.dataset = []
        for i, row in tqdm(split.iterrows()):
            reportFindings = findings["s" + str(row["study_id"]) + ".txt"]
            if reportFindings == "NO FINDINGS": continue

            with open(
                os.path.join(
                    self.path,
                    "files",
                    "p" + str(row["subject_id"])[:2],
                    "p" + str(row["subject_id"]),
                    "s" + str(row["study_id"]) + ".txt",
                ),
                "r",
            ) as f:
                text = f.read()
            report = self.parse_radiology_report(text)
            self.dataset.append(
                [
                    str(row["subject_id"]),
                    str(row["study_id"]),
                    str(row["dicom_id"]),
                    str(reportFindings),
                    str(report["INDICATION"]) if "INDICATION" in report else "",
                ]
            )

        self.dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(
                columns=[
                    "subject_id",
                    "study_id",
                    "dicom_id",
                    "findings",
                    "indication",
                ],
                data=self.dataset,
            )
        )

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        answersLog = []

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

            correctAnswers = [[self.getCorrectAnswer(sample)] for sample in batch]

            for idx, answer in enumerate(answers):
                answersLog.append((self.getCorrectAnswer(batch[idx]), answer))

                # Compute the number of tokens recalled in the answer
                # tokens = set(self.cleanStr(answer).split(" "))
                # correctTokens = set(self.cleanStr(self.getCorrectAnswer(batch[idx])).split(" "))
                # precision = len(tokens.intersection(correctTokens)) / len(tokens)
                # recall = len(tokens.intersection(correctTokens)) / len(correctTokens)
                # self.f1.append(2 * (precision * recall) / (precision + recall + 1e-8))
                self.f1.append(0)

            self.bleu_1.update(answers, correctAnswers)
            self.bleu_4.update(answers, correctAnswers)
            self.rougeL.update(answers, correctAnswers)

        # TODO: add others metrics such as AUC, F1...
        metrics = {
            "bleu1": self.bleu_1.compute(),
            "bleu4": self.bleu_4.compute(),
            "rougeL": self.rougeL.compute(),
            "f1": sum(self.f1) / len(self.f1),
        }

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def parse_radiology_report(self, report: str):
        # Split the report into lines
        lines = report.split("\n")

        # Initialize an empty dictionary
        radiology_dict = {}

        # Define the keys to look for
        keys_to_find = [
            "INDICATION",
            "COMPARISON",
            "TECHNIQUE",
            "FINDINGS",
            "IMPRESSION",
        ]

        currentField = None
        # Iterate through each line in the report
        for line in lines:
            # Split the line into key and value using the first colon
            parts = line.split(":", 1)

            # Check if the line has a colon and if the first part is a key we are interested in
            if len(parts) == 2 and remove_punctuation(parts[0].strip()) in keys_to_find:
                currentField = remove_punctuation(parts[0].strip())
                # Add the key-value pair to the dictionary
                radiology_dict[currentField] = parts[1].strip()
                continue

            if currentField is not None:
                radiology_dict[currentField] = radiology_dict[currentField] + line

        for key in radiology_dict:
            radiology_dict[key] = radiology_dict[key].strip()

        return radiology_dict

    def format_question(self, sample):
        imagePath = os.path.join(
            self.path,
            "files",
            "p" + str(sample["subject_id"])[:2],
            "p" + str(sample["subject_id"]),
            "s" + str(sample["study_id"]),
            sample["dicom_id"] + ".jpg",
        )

        if sample["indication"] == "":
            question = "Given <img>, what are the findings?"
        else:
            question = f"Given <img> and the following indications:\n {sample['indication']}\nWhat are the findings?"

        formattedText = [
            {
                "role": "user",
                "content": question,
            },
            {"role": "assistant", "content": f"Findings: {sample['findings']}"},
        ]

        return (formattedText, [Image.open(imagePath)])

    def getCorrectAnswer(self, sample):
        return sample["findings"]
