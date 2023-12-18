from multimedbench.utils import Benchmark
import json
import os
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import datasets


class MIMIC_III(Benchmark):
    def __init__(self, engine, **kwargs) -> None:
        super().__init__(engine, **kwargs)
        self.taskName = "MIMIC-III"

        self.path = json.load(open("MedMD_config.json", "r"))["MIMIC-III"]["path"]

        df = pd.read_csv(os.path.join(self.path, "NOTEEVENTS.csv"), low_memory=False)
        # Get all the differetn values for the column "category"

        # Keep only the Radiology reports
        df = df[df["CATEGORY"] == "Radiology"]

        goodReports = []
        for i in tqdm(range(df.shape[0])):
            parsedReport = self._parseReport(df.iloc[i]["TEXT"])
            if parsedReport is not None:
                goodReports.append(parsedReport)

        # Create a histogram of IMPRESSIONS and FINDINGS lengths
        # lengthImpression = []
        # lengthFindings = []
        # for report in goodReports:
        #     lengthImpression.append(len(report["impression"]))
        #     lengthFindings.append(len(report["findings"]))

        # plt.hist(lengthImpression, bins=100, alpha=0.5, label='Impression')
        # plt.hist(lengthFindings, bins=100, alpha=0.5, label='Findings')
        # plt.legend(loc='upper right')
        # plt.xlabel('Length')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Impression and Findings Lengths')
        # plt.savefig("impression_findings.png")

        self.dataset = datasets.Dataset.from_list(goodReports)

        print(self.dataset)

    
    def run(self, params, batcher):
        pass

    


    def _parseReport(self, report: str):
        try:
            # Find the "FINAl REPORT" and cut what's before
            report: str = report[report.index("FINAL REPORT") + len("FINAL REPORT") :]

        except Exception as e:
            return None

        report = [line.replace("\n", " ").strip() for line in report.split("\n\n")]

        sections = {}
        # Loop through the lines on the report. In the line starts with a word in capital and a column, it's a new section
        for line in report:
            currentSection = None
            if line == "":
                continue
            splitline = line.split(":")
            if len(splitline) > 1 and splitline[0].isupper():
                currentSection = splitline[0]
                sections[currentSection] = ""
                line = " ".join(splitline[1:])

            if currentSection is not None:
                sections[currentSection] = sections[currentSection] + " " + line

        # Strip and remove newlines in the sections
        for section in sections:
            sections[section] = sections[section].replace("\n", " ").strip()

        if "IMPRESSION" not in sections or "FINDINGS" not in sections or sections["IMPRESSION"] == "" or sections["FINDINGS"] == "":
            return None

        return {"impression": sections["IMPRESSION"], "findings": sections["FINDINGS"]}
