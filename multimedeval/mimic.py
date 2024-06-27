import os
import subprocess
from zipfile import ZipFile

import datasets
import pandas as pd
from PIL import Image

from multimedeval.taskFamilies import ReportComparison
from multimedeval.utils import remove_punctuation, section_text


class MIMIC_CXR_reportgen(ReportComparison):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.taskName = "MIMIC-CXR Report Generation"
        self.modality = "X-Ray"
        self.task = "Report Generation"

    def setup(self):
        # Get the dataset ====================================================================
        self.path = self.engine.getConfig()["MIMIC_CXR_dir"]

        if self.path is None:
            raise ValueError("The path to the MIMIC-CXR dataset is not set")

        self._generate_dataset()

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        # For each sample in the dataset, open its report and check if it contains the keyword "findings"
        self.dataset = []
        self.studyToDicoms = {}
        for _, row in split.iterrows():
            samplePath = os.path.join(
                self.path,
                "files",
                "p" + str(row["subject_id"])[:2],
                "p" + str(row["subject_id"]),
            )
            with open(
                os.path.join(samplePath, "s" + str(row["study_id"]) + ".txt"), "r"
            ) as f:
                report, categories, _ = section_text(f.read())

            if "findings" not in categories:
                continue

            reportFindings = report[categories.index("findings")]
            reportIndication = (
                report[categories.index("indication")]
                if "indication" in categories
                else ""
            )

            self.dataset.append(
                [
                    str(row["subject_id"]),
                    str(row["study_id"]),
                    str(reportFindings),
                    str(reportIndication),
                ]
            )

            if str(row["study_id"]) not in self.studyToDicoms:
                self.studyToDicoms[str(row["study_id"])] = [str(row["dicom_id"])]
            else:
                self.studyToDicoms[str(row["study_id"])].append(str(row["dicom_id"]))

        # Convert the dataset to a pandas dataframe
        self.dataset = pd.DataFrame(
            columns=["subject_id", "study_id", "findings", "indications"],
            data=self.dataset,
        )
        self.dataset = self.dataset.drop_duplicates()
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

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

    def format_question(self, sample, include_indication=False):

        samplePath = os.path.join(
            self.path,
            "files",
            "p" + sample["subject_id"][:2],
            "p" + sample["subject_id"],
        )

        dicomIndices = self.studyToDicoms[sample["study_id"]]

        imagesPath = [
            os.path.join(samplePath, "s" + sample["study_id"], dicomIndex + ".jpg")
            for dicomIndex in dicomIndices
        ]

        # indication = sample["indications"].strip().replace('\n', ' ').replace('  ', ' ')

        imgTags = "<img> " * len(imagesPath)

        question = (
            (sample["indications"] + " ")
            if ("indications" in sample and include_indication)
            else ""
        )
        question += (
            f"Can you provide a radiology report for this medical image? {imgTags}"
        )

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

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(
                self.path, "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-split.csv"
            )
        ):
            self.path = os.path.join(self.path, "mimic-cxr-jpg", "2.0.0")
            return

        os.makedirs(self.path, exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        wget_command = f'wget -r -c -np -nc --directory-prefix "{self.path}" --user "{username}" --password "{password}" https://physionet.org/files/mimic-cxr-jpg/2.0.0/'  # Can replace -nc (no clobber) with -N (timestamping)

        subprocess.run(wget_command, shell=True, check=True)

        self.path = os.path.join(self.path, "mimic-cxr-jpg", "2.0.0")

        # Unzip the mimic-cxr-2.0.0-split file
        file = os.path.join(self.path, "mimic-cxr-2.0.0-split.csv")
        with ZipFile(file + ".gz", "r") as zipObj:
            zipObj.extractall(file)
