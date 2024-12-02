"""CT-RATE Report Generation Task."""

import os
import datasets
import pandas as pd
from multimedeval.task_families import ReportComparison
from multimedeval.utils import BatcherInput
from huggingface_hub import hf_hub_download
import nibabel as nib
from tqdm import tqdm


class CTRATEReportGen(ReportComparison):
    """CT-RATE Report Generation Task."""

    def __init__(self, **kwargs):
        """Initialize the CT-RATE Report Generation Task."""
        super().__init__(**kwargs)

        self.task_name = "CT-RATE Report Generation"
        self.modality = "CT"
        self.task = "Report Generation"
        self.path = None

    def setup(self):
        """Setup the CT-RATE Report Generation Task."""
        self.path = self.engine.get_config()["ctrate_dir"]

        if self.path is None:
            raise ValueError("The path to the CT-RATE dataset is not set")

        self.hf_token = self.engine.get_config()["hf_token"]

        self._generate_dataset()

        dataframe = pd.read_csv(
            os.path.join(self.path, "radiology_text_reports", "validation_reports.csv")
        )
        data_directory_name = os.path.join(self.path, "valid")

        def convert_to_absolute_path(path: str):
            """
            Convert VolumeName into full absolute file path.
            """
            folder1 = path.split("_")[0]
            folder2 = path.split("_")[1]
            folder = folder1 + "_" + folder2
            folder3 = path.split("_")[2]
            subfolder = folder + "_" + folder3
            subfolder = os.path.join(data_directory_name, folder, subfolder)

            full_path = os.path.join(subfolder, path)

            return full_path

        dataframe["AbsoluteDataPath"] = dataframe["VolumeName"].apply(
            convert_to_absolute_path
        )
        dataframe = dataframe.drop_duplicates()
        self.dataset = datasets.Dataset.from_pandas(dataframe)

    def format_question(self, sample, prompt=False, include_indication=False):
        """Format the question for the user and the assistant.

        Args:
            sample: The sample to format.
            include_indication: Whether to include the indication section of \
                the report in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted prompt and the images.
        """
        sample_path = sample["AbsoluteDataPath"]

        question = (sample["Impressions_EN"] + " ") if (include_indication) else ""

        images = []
        if sample_path.endswith(".nii.gz"):
            image = nib.load(sample_path)
            images.append(image)

        question += (
            "Can you provide a radiology report for this set of CT scans? "
            + "<img>" * len(images)
        )

        batcher_input = BatcherInput()
        batcher_input._add_text_prompt("user", question)
        batcher_input._add_images(image=images)
        if prompt:
            batcher_input._add_text_prompt(
                "assistant", f"Findings: {sample['Findings_EN']}"
            )

        return batcher_input

    def get_correct_answer(self, sample):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.

        Returns:
            The correct answer.
        """
        return sample["Findings_EN"]

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        self.path = os.path.join(self.path, "ct_rate")

        os.makedirs(self.path, exist_ok=True)

        repo_id = "ibrahimhamamci/CT-RATE"
        config_directory_name = "dataset/radiology_text_reports"

        if not os.path.exists(
            os.path.join(
                self.path, "dataset", "radiology_text_reports", "validation_reports.csv"
            )
        ):
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                subfolder=config_directory_name,
                filename="validation_reports.csv",
                local_dir=self.path,
            )

        config_data = pd.read_csv(
            os.path.join(
                self.path, "dataset", "radiology_text_reports", "validation_reports.csv"
            )
        )

        if not self.hf_token:  # TODO change to the engine function
            if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
                self.hf_token = os.environ["HF_TOKEN"]
            raise ValueError(
                "Please include the personal hf_token in the SetupParams OR in the path variable."
            )

        directory_name = "dataset/valid/"
        for name in tqdm(config_data["VolumeName"], desc="Dowloading CT-RATE"):
            folder1 = name.split("_")[0]
            folder2 = name.split("_")[1]
            folder = folder1 + "_" + folder2
            folder3 = name.split("_")[2]
            subfolder = folder + "_" + folder3
            subfolder = directory_name + folder + "/" + subfolder
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                token=self.hf_token,
                subfolder=subfolder,
                filename=name,
                local_dir=self.path,
                resume_download=True,
            )

        self.path = os.path.join(self.path, "dataset")
