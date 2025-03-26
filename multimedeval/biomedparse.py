import os
import datasets
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from multimedeval.task_families import Segmentation
from multimedeval.utils import BatcherInput
from huggingface_hub import hf_hub_download
import zipfile

import numpy as np
from typing import List, Union
import json
import random

NAME_TO_BIOMEDPARSE = {
    "PathBiomedParse": {"modality": "Pathology"},
    "RadBiomedParse": {"modality": "X-Ray", "num_sample": 500},
    "EndoBiomedParse": {"modality": "Endoscopy", "num_sample": None},
    "DermaBiomedParse": {"modality": "Dermatology", "num_sample": None},
    "MRI_FS_BiomedParse": {"modality": "MRI", "num_sample": 2000},
    "MRI_HS_BiomedParse": {"modality": "MRI", "num_sample": 2000},
    "CTBiomedParse": {"modality": "CT", "num_sample": 1000},
}


class BiomedParse(Segmentation):
    """BiomedParse Segmentation task family."""

    def __init__(self, biomedparse_name, **kwargs):
        """Initialize the BiomedParse Segmentation task."""
        super().__init__(**kwargs)
        self.modality = NAME_TO_BIOMEDPARSE[biomedparse_name]["modality"]
        self.sample_size = NAME_TO_BIOMEDPARSE[biomedparse_name]["num_sample"]
        self.task_name = biomedparse_name

        self.dataset_file_names = None

    def setup(self):
        self.fewshot_counter = 0
        """Setup the BiomedParse Segmentation task."""
        self.path = self.engine.get_config()["biomedparse_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping Biomedparse because the cache directory is not set."
            )

        self._generate_dataset()

        concat_dataset = []  # (label, image_path, mask_path)
        for dataset in self.dataset_file_names:
            samples = self._post_process(dataset)
            concat_dataset.append(samples)

        self.dataset = pd.concat(concat_dataset, ignore_index=True)
        if self.sample_size is not None and isinstance(self.sample_size, int):
            self.dataset = self.dataset.sample(n=self.sample_size, random_state=42)
        # print(self.modality, len(self.dataset))
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

    def _post_process(self, dataset):
        """
        For the given dataset, convert the data into pd.DataFrame(label, image_path, seg_path)

        Return:
            pd.DataFrame[labels, abs_img_path, abs_seg_path].
        """
        img_folder, seg_folder, config = None, None, None
        data_folder = os.path.join(self.path, dataset)
        for root, dirs, files in os.walk(data_folder):
            for folder_name in dirs:
                if folder_name == "test":
                    img_folder = os.path.join(root, folder_name)
                elif folder_name == "test_mask":
                    seg_folder = os.path.join(root, folder_name)
            if "test.json" in files:
                config = os.path.join(root, "test.json")
        if img_folder and seg_folder and config:
            config_path = os.path.join(data_folder, config)
            with open(config_path, "r") as file:
                config_json = json.load(file)
            df = pd.json_normalize(config_json["annotations"])

            # BioMedParse provides n different prompts, here a random one is selected
            df["labels"] = df["sentences"].apply(lambda x: random.choice(x)["sent"])
            df["abs_seg_path"] = df["mask_file"].apply(
                lambda x: os.path.join(data_folder, seg_folder, x)
            )
            df["abs_img_path"] = df["file_name"].apply(
                lambda x: os.path.join(data_folder, img_folder, x)
            )
            df_selected = df[["labels", "abs_img_path", "abs_seg_path"]]
            return df_selected

    def get_predicted_answer(self, answer: Union[List[np.array]]):
        """Convert the predicted mask to one-hot encoding.

        Args:
            answer: The predicted segmentation mask.

        Returns:
            The one-hot encoded segmentation mask.
        """

        return answer[0]

    def get_correct_answer(self, sample):
        """Returns the ground truth mask for the sample.

        Args:
            sample: The sample to get the correct mask from.

        Returns:
            The one-hot encoded ground truth mask.
        """
        gt_mask = Image.open(sample["abs_seg_path"], formats=["png"])
        gt_mask_np = 1 * (np.array(gt_mask.convert("RGB"))[:, :, 0] > 0).astype("uint8")
        return gt_mask_np

    def format_question(self, sample, prompt=False):
        """Formats the question.

        Args:
            sample: The sample to format.
            prompt: Adds the answer to the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted prompt,
              images, and segmentation mask.
        """
        batcher_input = BatcherInput()

        question = f"<img> Segment {sample['labels']} "
        batcher_input._add_text_prompt("user", question)

        if prompt:
            batcher_input._add_text_prompt(
                "assistant", f"<seg{self.fewshot_counter % 5}>"
            )
            batcher_input._add_segmentation_mask(self.get_correct_answer(sample))
            self.fewshot_counter += 1

        image = Image.open(sample["abs_img_path"])
        batcher_input._add_images(image)
        return batcher_input

    def get_all_labels(self):
        return []

    def _generate_dataset(self):
        """
        Generate datasets through Huggingface, Data size: about 50 GB.
        """
        for file in self.dataset_file_names:
            if os.path.exists(os.path.join(self.path, file)):
                continue
            else:
                hf_hub_download(
                    repo_id="microsoft/BiomedParseData",
                    repo_type="dataset",
                    filename=file + ".zip",
                    cache_dir=self.path,
                    local_dir=self.path,
                )

            # Unzip the file
            with zipfile.ZipFile(
                os.path.join(self.path, file + ".zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(self.path)


class RadBiomedParse(BiomedParse):
    """BiomedParse Radiology task."""

    def __init__(self, **kwargs):
        super().__init__("RadBiomedParse", **kwargs)

        self.dataset_file_names = [
            "CXR_Masks_and_Labels",
            "COVID-QU-Ex",
            "siim-acr-pneumothorax",
            "QaTa-COV19",
        ]
        self.dataset_file_names += [
            os.path.join("Radiography", folder)
            for folder in ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
        ]


class PathBiomedParse(BiomedParse):
    """BiomedParse Pathology task."""

    def __init__(self, **kwargs):
        super().__init__("PathBiomedParse", **kwargs)

        self.dataset_file_names = ["GlaS", "PanNuke"]


class DermaBiomedParse(BiomedParse):
    """BiomedParse Dermatology task."""

    def __init__(self, **kwargs):
        super().__init__("DermaBiomedParse", **kwargs)

        self.dataset_file_names = ["ISIC", "UWaterlooSkinCancer"]


class EndoBiomedParse(BiomedParse):
    """BiomedParse Endoscopy task."""

    def __init__(self, **kwargs):
        super().__init__("EndoBiomedParse", **kwargs)

        self.dataset_file_names = ["PolypGen", "NeoPolyp"]


class CTBiomedParse(BiomedParse):
    """BiomedParse CT task."""

    def __init__(self, **kwargs):
        super().__init__("CTBiomedParse", **kwargs)

        self.dataset_file_names = [
            "COVID-19_CT",
            "LIDC-IDRI",
        ]
        self.dataset_file_names += [
            os.path.join(dataset_name, folder_name)
            for dataset_name, folder_name in [
                ("amos22", "CT"),
                ("MSD", "Task03_Liver"),
                ("MSD", "Task07_Pancreas"),
            ]
        ]


class MRI_FS_BiomedParse(BiomedParse):
    """BiomedParse Full-Sample MRI task."""

    def __init__(self, **kwargs):
        super().__init__("MRI_FS_BiomedParse", **kwargs)

        self.dataset_file_names = ["ACDC", "MMs", os.path.join("MSD", "Task02_Heart")]


class MRI_HS_BiomedParse(BiomedParse):
    """BiomedParse Half-Sample MRI task."""

    def __init__(self, **kwargs):
        super().__init__("MRI_HS_BiomedParse", **kwargs)
        self.dataset_file_names = ["LGG"]
        self.dataset_file_names += [
            os.path.join(dataset_name, folder_name)
            for dataset_name, folder_name in [
                ("amos22", "MRI"),
                ("MSD", "Task04_Hippocampus"),
                ("MSD", "Task01_BrainTumour"),
            ]
        ]
