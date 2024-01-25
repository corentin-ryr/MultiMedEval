import logging
import os
from multimedbench.utils import Benchmark
import pandas as pd
from tqdm import tqdm
from PIL import Image
import datasets
from multimedbench.chexbert.label import encode
from bert_score import BERTScorer

from multimedbench.utils import (
    Benchmark,
    Params,
    remove_punctuation,
    exact_entity_token_if_rel_exists_reward,
    section_text,
)
from torchmetrics.text import BLEUScore, ROUGEScore

import torch
import dill
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from zipfile import ZipFile
import requests
from requests.auth import HTTPBasicAuth
from torch.utils.data import DataLoader
import subprocess


class MIMIC_CXR_reportgen(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.debug("***** Transfer task : MIMIC_CXR *****\n\n")

        self.taskName = "MIMIC_CXR report generation"
        self.modality = "X-Ray"
        self.task = "Report generation"

        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys="rougeL")

        self.chexbertPath = self.engine.getConfig()["CheXBert"]["dlLocation"]

        # Get the dataset ====================================================================
        self.path = self.engine.getConfig()["physionet"]["path"]
        self._generate_dataset()

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        # For each sample in the dataset, open its report and check if it contains the keyword "findings"
        self.dataset = []
        self.studyToDicoms = {}
        for _, row in split.iterrows():
            samplePath = os.path.join(
                self.path, "files", "p" + str(row["subject_id"])[:2], "p" + str(row["subject_id"])
            )
            with open(os.path.join(samplePath, "s" + str(row["study_id"]) + ".txt"), "r") as f:
                report, categories, _ = section_text(f.read())

            if "findings" not in categories:
                continue

            reportFindings = report[categories.index("findings")]
            reportIndication = report[categories.index("indication")] if "indication" in categories else ""

            self.dataset.append(
                [str(row["subject_id"]), str(row["study_id"]), str(reportFindings), str(reportIndication)]
            )

            if str(row["study_id"]) not in self.studyToDicoms:
                self.studyToDicoms[str(row["study_id"])] = [str(row["dicom_id"])]
            else:
                self.studyToDicoms[str(row["study_id"])].append(str(row["dicom_id"]))

        # Convert the dataset to a pandas dataframe
        self.dataset = pd.DataFrame(columns=["subject_id", "study_id", "findings", "indications"], data=self.dataset)

        # Remove the duplicates
        self.dataset = self.dataset.drop_duplicates()

        self.dataset = datasets.Dataset.from_pandas(self.dataset)

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")
        refReports = []
        hypReports = []
        bleu1Scores = []
        bleu4Scores = []
        rougeLScores = []

        # Run the batcher for all data split in chunks
        dataloader = DataLoader(self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x)
        for batch in tqdm(
            dataloader,
            desc="Generating reports",
        ):
            batcherCorrect = [self.getCorrectAnswer(sample) for sample in batch]
            batcherHyp = batcher([self.format_question(sample) for sample in batch])
            batcherHyp = [h if h != "" else "Invalid Response" for h in batcherHyp]

            refReports += batcherCorrect
            hypReports += batcherHyp

            for hyp, ref in zip(batcherHyp, batcherCorrect):
                bleu1Scores.append(self.bleu_1([hyp], [[ref]]).item())
                bleu4Scores.append(self.bleu_4([hyp], [[ref]]).item())
                rougeLScores.append(self.rougeL([hyp], [[ref]])["rougeL_fmeasure"].item())

            # break

        f1_bertscore = self.compute_bertscore(hypReports, refReports)

        chexbert_similarity = self.compute_chexbert(hypReports, refReports)

        f1_radgraph = self.compute_radgraph(hypReports, refReports)

        bleu_scores = torch.tensor(
            [self.bleu_1([candidate], [[reference]]).item() for reference, candidate in zip(refReports, hypReports)]
        )

        radcliq_v0_scores = self.compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph)

        meteor_scores = self.compute_meteor(hypReports, refReports)

        rougeScores = self.rougeL.compute()
        rougeScores = {key: value.item() for key, value in rougeScores.items()}

        metrics = {
            "bleu1": self.bleu_1.compute().item(),
            "bleu4": self.bleu_4.compute().item(),
            "f1-radgraph": f1_radgraph.mean().item(),
            "CheXBert vector similarity": chexbert_similarity.mean().item(),
            "f1-bertscore": f1_bertscore.mean().item(),
            "radcliq": sum(radcliq_v0_scores) / len(radcliq_v0_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
        }
        metrics.update(rougeScores)

        answersLog = zip(refReports, hypReports, bleu1Scores, bleu4Scores, rougeLScores)
        # Add a header to the log
        answersLog = [("ref", "hyp", "bleu1", "bleu4", "rougeL")] + list(answersLog)

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
        samplePath = os.path.join(self.path, "files", "p" + sample["subject_id"][:2], "p" + sample["subject_id"])

        dicomIndices = self.studyToDicoms[sample["study_id"]]

        imagesPath = [
            os.path.join(samplePath, "s" + sample["study_id"], dicomIndex + ".jpg") for dicomIndex in dicomIndices
        ]

        # indication = sample["indications"].strip().replace('\n', ' ').replace('  ', ' ')

        imgTags = "<img> " * len(imagesPath)

        question = f"{imgTags}Please caption this scan with findings and impression."

        formattedText = [
            {
                "role": "user",
                "content": question,
            },
            # {"role": "assistant", "content": f"Findings: {sample['findings']}"},
        ]

        return (formattedText, [Image.open(imagePath) for imagePath in imagesPath])

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
        # Get the current path to the module
        module_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(module_path, "composite_metric_model_dill.pkl"), "rb") as f:
            composite_metric_v0_model = dill.load(f)

        with open(os.path.join(module_path, "normalizer_dill.pkl"), "rb") as f:
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

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(os.path.join(self.path, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-split.csv")):
            self.path = os.path.join(self.path, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0")
            return

        os.makedirs(self.path, exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        wget_command = f'wget -r -c -np -nc --directory-prefix "{self.path}" --user "{username}" --password "{password}" https://physionet.org/files/mimic-cxr-jpg/2.0.0/' # Can replace -nc (no clobber) with -N (timestamping)

        subprocess.run(wget_command, shell=True, check=True)
        
        self.path = os.path.join(self.path, "physionet.org", "files", "mimic-cxr-jpg", "2.0.0")

        # Unzip the mimic-cxr-2.0.0-split file
        file = os.path.join(self.path, "mimic-cxr-2.0.0-split.csv")
        with ZipFile(file + ".gz", "r") as zipObj:
            zipObj.extractall(file)
