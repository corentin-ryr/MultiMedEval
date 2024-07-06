"""MIMIC-CXR Report Generation Task."""

import os
import subprocess
from zipfile import ZipFile

import datasets
import pandas as pd
from PIL import Image

from multimedeval.task_families import ReportComparison
from multimedeval.utils import section_text


class MIMICCXRReportgen(ReportComparison):
    """MIMIC-CXR Report Generation Task."""

    def __init__(self, **kwargs):
        """Initialize the MIMIC-CXR Report Generation Task."""
        super().__init__(**kwargs)

        self.task_name = "MIMIC-CXR Report Generation"
        self.modality = "X-Ray"
        self.task = "Report Generation"
        self.path = None

    def setup(self):
        """Setup the MIMIC-CXR Report Generation Task."""
        # Get the dataset ====================================================================
        self.path = self.engine.getConfig()["mimic_cxr_dir"]

        if self.path is None:
            raise ValueError("The path to the MIMIC-CXR dataset is not set")

        self._generate_dataset()

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        # For each sample in the dataset, open its report and check if it
        # contains the keyword "findings"
        self.dataset = []
        self.study_to_dicoms = {}
        for _, row in split.iterrows():
            sample_path = os.path.join(
                self.path,
                "files",
                "p" + str(row["subject_id"])[:2],
                "p" + str(row["subject_id"]),
            )
            with open(
                os.path.join(sample_path, "s" + str(row["study_id"]) + ".txt"),
                "r",
                encoding="utf-8",
            ) as f:
                report, categories, _ = section_text(f.read())

            if "findings" not in categories:
                continue

            report_findings = report[categories.index("findings")]
            report_indication = (
                report[categories.index("indication")]
                if "indication" in categories
                else ""
            )

            self.dataset.append(
                [
                    str(row["subject_id"]),
                    str(row["study_id"]),
                    str(report_findings),
                    str(report_indication),
                ]
            )

            if str(row["study_id"]) not in self.study_to_dicoms:
                self.study_to_dicoms[str(row["study_id"])] = [str(row["dicom_id"])]
            else:
                self.study_to_dicoms[str(row["study_id"])].append(str(row["dicom_id"]))

        # Convert the dataset to a pandas dataframe
        self.dataset = pd.DataFrame(
            columns=["subject_id", "study_id", "findings", "indications"],
            data=self.dataset,
        )
        self.dataset = self.dataset.drop_duplicates()
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

    def format_question(self, sample, prompt=False, include_indication=False):
        """Format the question for the user and the assistant.

        Args:
            sample: The sample to format.
            include_indication: Whether to include the indication section of \
                the report in the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        sample_path = os.path.join(
            self.path,
            "files",
            "p" + sample["subject_id"][:2],
            "p" + sample["subject_id"],
        )

        dicom_indices = self.study_to_dicoms[sample["study_id"]]

        images_path = [
            os.path.join(sample_path, "s" + sample["study_id"], dicomIndex + ".jpg")
            for dicomIndex in dicom_indices
        ]

        # indication = sample["indications"].strip().replace('\n', ' ').replace('  ', ' ')

        img_tags = "<img> " * len(images_path)

        question = (
            (sample["indications"] + " ")
            if ("indications" in sample and include_indication)
            else ""
        )
        question += (
            f"Can you provide a radiology report for this medical image? {img_tags}"
        )

        formatted_text = [
            {
                "role": "user",
                "content": question,
            },
            # {"role": "assistant", "content": f"Findings: {sample['findings']}"},
        ]

        return (formatted_text, [Image.open(imagePath) for imagePath in images_path])

    def get_correct_answer(self, sample):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.

        Returns:
            The correct answer.
        """
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
        wget_command = (
            f'wget -r -c -np -nc --directory-prefix "{self.path}" '
            f'--user "{username}" --password "{password}" '
            "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
        )

        subprocess.run(wget_command, shell=True, check=True)

        self.path = os.path.join(self.path, "mimic-cxr-jpg", "2.0.0")

        # Unzip the mimic-cxr-2.0.0-split file
        file = os.path.join(self.path, "mimic-cxr-2.0.0-split.csv")
        with ZipFile(file + ".gz", "r") as zip_obj:
            zip_obj.extractall(file)
