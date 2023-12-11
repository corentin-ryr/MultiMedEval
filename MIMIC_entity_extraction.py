import os
import sys
import json
import gdown
import pandas as pd
import csv
import datasets
from tqdm import tqdm
import math
from multimedbench.utils import batchSampler


class MIMIC_entity_extraction:
    def __init__(self):
        self._prepare_radgraph()

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

    def _prepare_radgraph(self):
        # Open the MedMD_config json file and get the download location for radgraph
        with open("MedMD_config.json", "r") as f:
            output = json.load(f)["RadGraph"]["dlLocation"]

        if not os.path.exists(os.path.join(output, "scorers")):
            gdown.download("https://drive.google.com/uc?id=1koePS_rgP5_zNUeqnQgdQ89nQEolTEbR", output, quiet=False)

            # Unzip the archive and delete the archive
            import zipfile

            with zipfile.ZipFile(os.path.join(output, "scorers.zip"), "r") as zip_ref:
                zip_ref.extractall(os.path.join(output, "scorers"))
            os.remove(os.path.join(output, "scorers.zip"))
        else:
            print("RadGraph already downloaded")

        # Add the RadGraph to the path
        sys.path.append(output)

        try:
            from scorers.RadGraph.RadGraph import (
                RadGraph,
            )  # It is normal that the import is not found by the IDE because it will be downloaded and installed at runtime
        except Exception as e:
            print("There was an error during the download and install of RadGraph")
            raise e

        self.radgraph = RadGraph(reward_level="partial")

    def create_reports_graph(self, ref, hyp):
        # Compute the F1-radgraph score
        (_, _, hyp_annotation_lists, ref_annotation_lists) = self.radgraph(refs=[ref], hyps=[hyp])

        return hyp_annotation_lists[0]["entities"]

    def run(self):
        for batch in tqdm(
            batchSampler(self.dataset, 4),
            total=math.ceil(len(self.dataset) / 4),
            desc="Generating reports",
        ):
            refReports = [sample["findings"] for sample in batch]

            for report in refReports:
                report = "The lungs show signs of pneumonia."
                entities = self.create_reports_graph(report, report)

                print(entities)

                # self.dependency_parser(report)

                raise Exception

    def dependency_parser(self, report):
        import spacy
        from spacy import displacy

        # Load the language model
        nlp = spacy.load("en_core_web_sm")

        # nlp function returns an object with individual token information,
        # linguistic features and relationships
        doc = nlp(report)

        print("{:<15} | {:<8} | {:<15} | {:<20}".format("Token", "Relation", "Head", "Children"))
        print("-" * 70)

        for token in doc:
            # Print the token, dependency nature, head and all dependents of the token
            print(
                "{:<15} | {:<8} | {:<15} | {:<20}".format(
                    str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])
                )
            )

        # Use displayCy to visualize the dependency
        svg = displacy.render(doc, style="dep", options={"distance": 120})

        # Write the output to a file
        with open("dependency.svg", "w") as f:
            f.write(svg)


extractor = MIMIC_entity_extraction()
extractor.run()
