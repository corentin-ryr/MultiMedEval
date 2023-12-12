import logging
import os
from multimedbench.utils import Benchmark
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import datasets
from multimedbench.chexbert.label import encode
from bert_score import BERTScorer

from multimedbench.utils import (
    Benchmark,
    batchSampler,
    Params,
    remove_punctuation,
    exact_entity_token_if_rel_exists_reward,
    section_text
)
import math
from torchmetrics.text import BLEUScore, ROUGEScore

import torch
import dill
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("wordnet")


class MIMIC_CXR_reportgen(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.debug("***** Transfer task : MIMIC_CXR *****\n\n")

        self.taskName = "MIMIC_CXR report generation"

        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys="rougeL")

        self.chexbertPath = json.load(open("MedMD_config.json", "r"))["CheXBert"]["dlLocation"]

        # Get the dataset ====================================================================
        self.path = json.load(open("MedMD_config.json", "r"))["MIMIC-CXR"]["path"]

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        # For each sample in the dataset, open its report and check if it contains the keyword "findings"
        self.dataset = []
        for _, row in split.iterrows():
            samplePath = os.path.join(
                self.path, "files", "p" + str(row["subject_id"])[:2], "p" + str(row["subject_id"])
            )
            with open(os.path.join(samplePath, "s" + str(row["study_id"]) + ".txt"), "r") as f:
                report, categories, _ = section_text(f.read())

            if "findings" not in categories:
                continue
            
            reportFindings = report[categories.index("findings")]

            if "indication" in categories:
                reportIndication = report[categories.index("indication")]
            else:
                reportIndication = ""

            self.dataset.append(
                [str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"]), str(reportFindings), str(reportIndication)]
            )

        self.dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(columns=["subject_id", "study_id", "dicom_id", "findings", "indications"], data=self.dataset)
        )

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")
        refReports = []
        hypReports = []

        # Run the batcher for all data split in chunks
        for batch in tqdm(
            batchSampler(self.dataset, params.batch_size),
            total=math.ceil(len(self.dataset) / params.batch_size),
            desc="Generating reports",
        ):
            refReports += [self.getCorrectAnswer(sample) for sample in batch]
            hypReports += batcher([self.format_question(sample) for sample in batch])
            # break

        refReportsNested = [[report] for report in refReports]
        self.bleu_1.update(hypReports, refReportsNested)
        self.bleu_4.update(hypReports, refReportsNested)
        self.rougeL.update(hypReports, refReportsNested)

        f1_bertscore = self.compute_bertscore(hypReports, refReports)

        chexbert_similarity = self.compute_chexbert(hypReports, refReports)

        f1_radgraph = self.compute_radgraph(hypReports, refReports)

        bleu_scores = torch.tensor(
            [self.bleu_1([candidate], [[reference]]).item() for reference, candidate in zip(refReports, hypReports)]
        )

        radcliq_v0_scores = self.compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph)

        meteor_scores = self.compute_meteor(hypReports, refReports)

        metrics = {
            "bleu1": self.bleu_1.compute().item(),
            "bleu4": self.bleu_4.compute().item(),
            "rougeL": self.rougeL.compute(),
            "f1-radgraph": f1_radgraph.mean().item(),
            "CheXBert vector similarity": chexbert_similarity.mean().item(),
            "f1-bertscore": f1_bertscore.mean().item(),
            "radcliq": sum(radcliq_v0_scores) / len(radcliq_v0_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
        }

        answersLog = zip(refReports, hypReports)

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

        # with open(os.path.join(samplePath, "s" + str(sample["study_id"]) + ".txt"), "r") as f:
        #     report = self.parse_radiology_report(f.read())
        #     indication = report["INDICATION"] if "INDICATION" in report else ""
        
        indication = sample["indications"]

        if indication == "":
            question = "Given <img>, can you provide a radiology report for this medical image?"
        else:
            question = f"Given <img> and the following indications:\n {indication}\nCan you provide a radiology report for this medical image?"

        formattedText = [
            {
                "role": "user",
                "content": question,
            },
            # {"role": "assistant", "content": f"Findings: {sample['findings']}"},
        ]

        return (formattedText, [Image.open(imagePath)])

    def getCorrectAnswer(self, sample):
        return sample["findings"]

    def compute_chexbert(self, hypReports, refReports):
        df = pd.DataFrame(columns=["Report Impression"], data=refReports)
        labelsReference = encode(os.path.join(self.chexbertPath, "chexbert.pth"), df)

        df = pd.DataFrame(columns=["Report Impression"], data=hypReports)
        labelsHypothesis = encode(os.path.join(self.chexbertPath, "chexbert.pth"), df)

        # Compute the vector similarity between the reference and the geenrated reports
        return torch.cosine_similarity(labelsReference, labelsHypothesis)

    def compute_meteor(self, hypReports, refReports):
        meteor_scores = []
        for ref, hyp in zip(refReports, hypReports):
            # Tokenize the reference and hypothesis
            ref_tokens = word_tokenize(ref)
            hyp_tokens = word_tokenize(hyp)

            # Compute the meteor score
            meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))

        return meteor_scores

    def compute_composite(self, bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph):
        with open("multimedbench/composite_metric_model_dill.pkl", "rb") as f:
            composite_metric_v0_model = dill.load(f)

        with open("multimedbench/normalizer_dill.pkl", "rb") as f:
            normalizer = dill.load(f)

        # The column need to be in the order [bleu, bertscore, chexbert, radgraph]
        input_data = torch.stack([bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph], dim=1)

        norm_input_data = normalizer.transform(input_data)
        return composite_metric_v0_model.predict(norm_input_data)

    def compute_bertscore(self, hypReports, refReports):
        scorer = BERTScorer(
            model_type="distilroberta-base",
            batch_size=256,
            lang="en",
            rescale_with_baseline=True,
            idf=True,
            idf_sents=hypReports,
        )

        P, R, f1_bertscore = scorer.score(hypReports, refReports)
        return f1_bertscore

    def compute_radgraph(self, hypReports, refReports):
        f1_radgraph = []
        for hyp, ref in zip(hypReports, refReports):
            # Compute the F1-radgraph score
            (_, _, hyp_annotation_lists, ref_annotation_lists) = self.engine.radgraph(refs=[ref], hyps=[hyp])
            f1_radgraph.append(
                exact_entity_token_if_rel_exists_reward(hyp_annotation_lists[0], ref_annotation_lists[0])
            )
        return torch.tensor(f1_radgraph)
