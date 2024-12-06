import os
import datasets
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from multimedeval.task_families import Segmentation
from multimedeval.utils import BatcherInput

import numpy as np
from typing import List, Union


class REFUGE(Segmentation):
    """REFUGE Segmentation task."""

    def __init__(self, **kwargs):
        """Initialize the REFUGE Segmentation task."""
        super().__init__(**kwargs)
        self.modality = "Retinal Fundus Image"
        self.task_name = "REFUGE"
        self.class_mapping = {"Optic Disk": 128, "Optic Cup": 0}

    def setup(self):
        self.fewshot_counter = 0
        """Setup the REFUGE Segmentation task."""
        self.path = self.engine.get_config()["refuge_dir"]

        if self.path is None:
            raise ValueError("Skipping REFUGE because the cache directory is not set.")

        self._generate_dataset()

        ## Note: train_dataset refers to the "val" part of true dataset
        dataset_path = os.path.join(self.path, "test")
        val_dataset_path = os.path.join(self.path, "val")
        dataset_df = self._generate_config_dataset_df(dataset_path, "bmp")

        self.dataset = datasets.Dataset.from_pandas(dataset_df)
        self.train_dataset = datasets.Dataset.from_pandas(
            self._generate_config_dataset_df(val_dataset_path, "png")
        )

    def _generate_config_dataset_df(self, path, suffix):
        """
        Generate pd.DataFrame containing column information on
        image path, segmentation mask path and label.

        Args:
            path: path to the test/train/val partition of dataset.
            suffix: validation set uses png format for segmentation,
                    while test set uses bmp.

        Return:
            pd.dataFrame with three columns (img_path, seg_path, label)
        """
        config_list = []
        image_path = os.path.join(path, "images")
        seg_path = os.path.join(path, "mask")
        if os.path.exists(image_path) and os.path.exists(seg_path):
            for label in self.class_mapping.keys():
                for f in os.listdir(image_path):
                    if os.path.isfile(os.path.join(image_path, f)):
                        seg_file = f[:-3] + suffix
                        config_list.append(
                            [
                                os.path.join(image_path, f),
                                os.path.join(seg_path, seg_file),
                                label,
                            ]
                        )
            return pd.DataFrame(
                data=config_list, columns=["img_path", "seg_path", "label"]
            )
        else:
            raise ValueError(f"path invalid:{path}")

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

        return self._get_binary_mask(sample)

    def _get_binary_mask(self, sample):
        """Convert sample to binary mask according to sample["label"].

        Args:
            sample: The sample to get the correct mask from.

        Returns:
            The one-hot encoded ground truth mask.
        """
        pixel_value = self.class_mapping[sample["label"]]
        gt_mask_path = sample["seg_path"]
        gt_mask_np = np.array(Image.open(gt_mask_path))

        gt_mask_binary = (gt_mask_np == pixel_value).astype(int)

        return gt_mask_binary

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

        question = f"<img> Segment {sample['label']}."
        batcher_input._add_text_prompt("user", question)

        if prompt:
            batcher_input._add_text_prompt(
                "assistant", f"<seg{self.fewshot_counter % 5}>"
            )
            batcher_input._add_segmentation_mask(self._get_binary_mask(sample))
            self.fewshot_counter += 1

        image = Image.open(sample["img_path"])
        batcher_input._add_images(image)

        return batcher_input

    def get_all_labels(self):
        return list(self.class_mapping.keys())

    def _generate_dataset(self):
        """
        Generate datasets through Kaggle, Data size: about 10 GB.
        """
        if os.path.exists(os.path.join(self.path, "REFUGE2")):
            self.path = os.path.join(self.path, "REFUGE2")
            return

        api = KaggleApi()
        api.authenticate()

        os.makedirs(self.path, exist_ok=True)

        api.dataset_download_files("victorlemosml/refuge2", path=self.path, unzip=True)

        self.path = os.path.join(self.path, "REFUGE2")
