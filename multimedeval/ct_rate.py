"""CT-RATE Report Generation Task."""

import os
import datasets
import pandas as pd
from multimedeval.task_families import ReportComparison, ImageClassification
from multimedeval.utils import BatcherInput,clean_str
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

        self.hf_token = self.engine.get_huggingface_token()

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

class CTRATEClassification(ImageClassification):
    """CT-RATE Image Classification Task."""

    def __init__(self, **kwargs):
        """Initialize the CT-RATE Image Classification Task."""
        super().__init__(**kwargs)

        self.task_name = "CT-RATE Image Classification"
        self.modality = "CT"
        self.path = None

    def setup(self):
        """Setup the CT-RATE Report Generation Task."""
        self.path = self.engine.get_config()["ctrate_dir"]
        self.scoring_type = "multilabel"
        self.num_classes = 18

        if self.path is None:
            raise ValueError("The path to the CT-RATE dataset is not set")
        
        self.hf_token = self.engine.get_huggingface_token()

        self._generate_dataset()

        self.dataset = pd.read_csv(
            os.path.join(
                self.path, "multi_abnormality_labels", "valid_predicted_labels.csv")
                )
        self.options = list(self.dataset.columns)[1:]

        if len(self.options) != self.num_classes:
            raise ValueError("Number of options don't match with that of classes. Check if the correct configuration file is downloaed.")
        
        data_directory_name = os.path.join(self.path, "valid")

        def convert_to_absolute_path(path : str):
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
        
        self.dataset["AbsoluteDataPath"]  = self.dataset["VolumeName"].apply(convert_to_absolute_path)
        self.dataset = self.dataset.drop_duplicates()
        
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

    def format_question(self, sample, prompt=False):
        """Format the question for the user and the assistant.

        Args:
            sample: The sample to format.
            prompt: Adds the answer to the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted prompt and the images.
        """
        sample_path = sample["AbsoluteDataPath"]
        
        question = "<img> Options:\n"
        question += " \n ".join(
            [f"{option}" for option in self.options]
        )
        question += " \n List the options that can be seen in this set of CT scans."
         
        images = []
        if sample_path.endswith(".nii.gz"):
            image = nib.load(sample_path)
            images.append(image)

        batcher_input = BatcherInput()
        batcher_input._add_text_prompt("user", question)
        batcher_input._add_images(image = images)
        if prompt:
            batcher_input._add_text_prompt("assistant", f"{self.get_correct_answer(sample, full_text=True)}")

        return batcher_input

    def get_predicted_answer(self, answer):
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            A list of predicted indices of the answer, e.g. [1,1,0,...,0].
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[clean_str(option)]]) for option in self.options]

        # for each 1 if above a threshold, 0 otherwise,
        return [1 if score > 0.5 else 0 for score in scores]

    def get_correct_answer(self, sample, full_text=False):
        """Returns the indices of correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from. 
                    "label" may be a string with multiple labels separated by "|".
            full_text: Whether to return the full text of the answer. Defaults to False.

        Returns:
            The correct answer as a list of true indices or a full-text string.
        """

        if full_text:
            # Return the full text for each label
            return [key for key, value in sample.items() if value == 1]

        label = [1 if sample[i] == 1 else 0 for i in self.options]
        return label


    def _generate_dataset(self):
        # Check if the path already exists and if so return
        self.path = os.path.join(self.path, "ct_rate")

        os.makedirs(self.path, exist_ok=True)

        repo_id = "ibrahimhamamci/CT-RATE"
        config_directory_name = "dataset/multi_abnormality_labels"

        if not os.path.exists(
            os.path.join(
                self.path, "dataset","multi_abnormality_labels","valid_predicted_labels.csv"
            )
        ):
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                subfolder=config_directory_name,
                filename="valid_predicted_labels.csv",
                local_dir=self.path,
            )

        config_data = pd.read_csv(
            os.path.join(
                self.path, "dataset","multi_abnormality_labels","valid_predicted_labels.csv"
            )
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