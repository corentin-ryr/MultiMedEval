import logging
import os
from multimedbench.utils import Benchmark
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import datasets
import numpy as np
from multimedbench.chexbert.label import label, encode

import time
from multimedbench.utils import (
    Benchmark,
    batchSampler,
    Params,
    remove_punctuation,
    exact_entity_token_if_rel_exists_reward,
)
import math
from torchmetrics.text import BLEUScore, ROUGEScore

import csv
import torch


class MIMIC_CXR_reportgen(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.debug("***** Transfer task : MIMIC_CXR *****\n\n")

        self.taskName = "MIMIC_CXR report generation"

        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys="rougeL")

        self.f1 = []

        self.chexbertPath = json.load(open("MedMD_config.json", "r"))["CheXBert"]["dlLocation"]

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
        for _, row in split.iterrows():
            reportFindings = findings["s" + str(row["study_id"]) + ".txt"]
            if reportFindings == "NO FINDINGS" or reportFindings == "":
                continue

            self.dataset.append(
                [str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"]), str(reportFindings)]
            )

        self.dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(columns=["subject_id", "study_id", "dicom_id", "findings"], data=self.dataset)
        )

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")
        answersLog = []

        refReports = []
        hypReports = []

        # Run the batcher for all data split in chunks
        for batch in tqdm(
            batchSampler(self.dataset, params.batch_size),
            total=math.ceil(len(self.dataset) / params.batch_size),
            desc="Running inference",
        ):
            batchPrompts = [self.format_question(sample) for sample in batch]
            answers = batcher(batchPrompts)

            for idx, sample in enumerate(batch):
                # Compute the F1-radgraph score
                (mean_reward, _, hypothesis_annotation_lists, reference_annotation_lists) = self.engine.radgraph(
                    refs=[self.getCorrectAnswer(sample)], hyps=[answers[idx]]
                )
                self.f1.append(exact_entity_token_if_rel_exists_reward(hypothesis_annotation_lists[0], reference_annotation_lists[0]))
            
            refReports += [self.getCorrectAnswer(sample) for sample in batch]
            hypReports += answers

            refReportsNested = [[self.getCorrectAnswer(sample)] for sample in batch]
            self.bleu_1.update(answers, refReportsNested)
            self.bleu_4.update(answers, refReportsNested)
            self.rougeL.update(answers, refReportsNested)

            break

        df = pd.DataFrame(columns=["Report Impression"], data=refReports)
        labelsReference = encode(os.path.join(self.chexbertPath, "chexbert.pth"), df)

        df = pd.DataFrame(columns=["Report Impression"], data=hypReports)
        labelsHypothesis = encode(os.path.join(self.chexbertPath, "chexbert.pth"), df)

        # Compute the vector similarity between the reference and the geenrated reports
        similarity = torch.cosine_similarity(labelsReference, labelsHypothesis)
        
        # TODO: add others metrics such as AUC, F1...
        metrics = {
            "bleu1": self.bleu_1.compute().item(),
            "bleu4": self.bleu_4.compute().item(),
            "rougeL": self.rougeL.compute(),
            "f1-radgraph": sum(self.f1) / len(self.f1),
            "CheXBert vector similarity": similarity.mean().item(),
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
        keys_to_find = ["INDICATION", "COMPARISON", "TECHNIQUE", "FINDINGS", "IMPRESSION"]

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
        samplePath = os.path.join(
            self.path, "files", "p" + str(sample["subject_id"])[:2], "p" + str(sample["subject_id"])
        )

        imagePath = os.path.join(samplePath, "s" + str(sample["study_id"]), sample["dicom_id"] + ".jpg")

        with open(os.path.join(samplePath, "s" + str(sample["study_id"]) + ".txt"), "r") as f:
            report = self.parse_radiology_report(f.read())
            indication = report["INDICATION"] if "INDICATION" in report else ""

        if indication == "":
            question = "Given <img>, what are the findings?"
        else:
            question = f"Given <img> and the following indications:\n {indication}\nWhat are the findings?"

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

