"""VQA tasks."""

import os
from zipfile import ZipFile

import gdown
import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image

from multimedeval.task_families import VQA
from multimedeval.utils import download_file


class VQARad(VQA):
    """VQA-Rad task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the VQA-Rad task object."""
        super().__init__(**kwargs)
        self.task_name = "VQA-Rad"
        self.modality = "Radiology"

    def setup(self):
        """Setup the VQA-Rad task. Downloads the dataset if not already downloaded."""
        cache_dir = self.engine.get_config()["vqa_rad_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for VQA-Rad dataset provided in the config file. "
                "Skipping the task."
            )

        self.dataset = load_dataset(
            "flaviagiammarino/vqa-rad", split="test", cache_dir=cache_dir
        )
        self.train_dataset = load_dataset(
            "flaviagiammarino/vqa-rad", split="train", cache_dir=cache_dir
        )

    def format_question(self, sample, prompt=False):
        """Format the question for the VQA-Rad task.

        Args:
            sample: The dataset sample to format.
            prompt: Wether or not to add the answer to the formatted question. Defaults to False.

        Returns:
            A conversation in the huggingface style and the images.
        """
        formatted_question = f"<img> {sample['question']}"
        formatted_answer = f"{sample['answer']}"
        if formatted_answer in ["yes", "no"]:
            formatted_question = (
                "Answer the following question with yes or no. " + formatted_question
            )

        question = [{"role": "user", "content": formatted_question}]
        if prompt:
            question.append({"role": "assistant", "content": formatted_answer})

        return (question, [sample["image"]])

    def get_correct_answer(self, sample):
        """Get the correct answer for the VQA-Rad task.

        Args:
            sample: The dataset sample.

        Returns:
            The correct answer.
        """
        return sample["answer"].lower().strip()


class PathVQA(VQA):
    """Path-VQA task."""

    def __init__(self, **kwargs):
        """Initialize the Path-VQA task object."""
        super().__init__(**kwargs)
        self.task_name = "VQA-Path"
        self.modality = "Pathology"

    def setup(self):
        """Setup the Path-VQA task. Downloads the dataset if not already downloaded."""
        cache_dir = self.engine.get_config()["path_vqa_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for Path-VQA dataset provided in the config file. Skipping the task."
            )

        self.dataset = load_dataset(
            "flaviagiammarino/path-vqa", split="test", cache_dir=cache_dir
        )
        self.train_dataset = load_dataset(
            "flaviagiammarino/path-vqa", split="train", cache_dir=cache_dir
        )

    def format_question(self, sample, prompt=False):
        """Format the question for the Path-VQA task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer to the formatted output. Defaults to False.

        Returns:
            A conversation in the huggingface style and the images.
        """
        formatted_question = f"<img> {sample['question']}"
        formatted_answer = f"{sample['answer']}"
        if formatted_answer in ["yes", "no"]:
            formatted_question = (
                "Answer the following question with yes or no. " + formatted_question
            )

        question = [{"role": "user", "content": formatted_question}]
        if prompt:
            question.append({"role": "assistant", "content": formatted_answer})

        return (question, [sample["image"]])

    def get_correct_answer(self, sample):
        """Get the correct answer for the Path-VQA task.

        Args:
            sample: The sample to get the correct answer for.

        Returns:
            The correct answer.
        """
        return sample["answer"].lower().strip()


class SLAKE(VQA):
    """VQA task for the SLAKE dataset."""

    def __init__(self, **kwargs):
        """Initialize the SLAKE task object."""
        super().__init__(**kwargs)
        self.task_name = "SLAKE"
        self.modality = "Radiology"

    def setup(self):
        """Setup the SLAKE task. Downloads the dataset if not already downloaded."""
        self.path = self.engine.get_config()["slake_dir"]

        if self.path is None:
            raise ValueError(
                "No path for SLAKE dataset provided in the config file. "
                "Skipping the task."
            )

        dset = load_dataset("BoKelvin/SLAKE", cache_dir=self.path)
        self._generate_dataset()

        self.path = os.path.join(self.path, "BoKelvin___slake")

        # jsonFile = json.load(open(os.path.join(self.path, "test.json"), "r"))
        self.dataset = []
        for sample in dset["test"]:
            if sample["q_lang"] == "en":
                # Open the image
                sample["image"] = os.path.join(self.path, "imgs", sample["img_name"])
                self.dataset.append(sample)

        # jsonFile = json.load(open(os.path.join(self.path, "train.json"), "r"))
        self.train_dataset = []
        for sample in dset["train"]:
            if sample["q_lang"] == "en":
                sample["image"] = os.path.join(self.path, "imgs", sample["img_name"])
                self.train_dataset.append(sample)

        self.dataset = Dataset.from_list(self.dataset)
        self.train_dataset = Dataset.from_list(self.train_dataset)

    def format_question(self, sample, prompt=False):
        """Format the question for the SLAKE task.

        Args:
            sample: The sample to format.
            prompt: Whether to add the answer to the conversation. Defaults to False.

        Returns:
            A conversation in the huggingface style and the images.
        """
        formatted_question = f"<img> {sample['question']}"
        formatted_answer = f"{sample['answer']}"
        if formatted_answer in ["yes", "no"]:
            formatted_question = (
                "Answer the following question with yes or no. " + formatted_question
            )

        question = [{"role": "user", "content": formatted_question}]
        if prompt:
            question.append({"role": "assistant", "content": formatted_answer})

        images = [Image.open(os.path.join(self.path, "imgs", sample["img_name"]))]

        return (question, images)

    def get_correct_answer(self, sample):
        """Get the correct answer for the SLAKE task.

        Args:
            sample: The sample to get the correct answer for.

        Returns:
            The correct answer.
        """
        return sample["answer"].lower().strip()

    def _generate_dataset(self):
        # Check if the path specified contains a SLAKE folder
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        else:
            if os.path.exists(os.path.join(self.path, "BoKelvin___slake", "imgs")):
                return

        output = os.path.join(self.path, "BoKelvin___slake", "imgs.zip")
        if not os.path.exists(output):
            gdown.download(
                "https://huggingface.co/datasets/BoKelvin/"
                "SLAKE/resolve/main/imgs.zip?download=true",
                output=output,
                quiet=False,
            )

        print(output)
        with ZipFile(output, "r") as z_object:
            z_object.extractall(path=os.path.join(self.path, "BoKelvin___slake"))

        os.remove(output)


class DiffVQA(VQA):
    """VQA task for the DiffVQA dataset."""

    def __init__(self, **kwargs):
        """Initialize the DiffVQA task object."""
        super().__init__(**kwargs)
        self.task_name = "Diff-VQA"
        self.modality = "Radiology"

    def setup(self):
        """Setup the DiffVQA task. Downloads the dataset if not already downloaded."""
        self.path = self.engine.get_config()["diff_vqa_dir"]

        self.mimic_path = self.engine.get_config()["mimic_cxr_dir"]

        if self.path is None or self.mimic_path is None:
            raise ValueError(
                "No path for DiffVQA dataset provided in the config file. Skipping the task."
            )

        self._generate_dataset()

        # Open the csv file
        df = pd.read_csv(os.path.join(self.path, "mimic_pair_questions.csv"))

        test_df = df[df["split"] == "test"]
        self.dataset = Dataset.from_pandas(test_df)

        train_df = df[df["split"] == "train"]
        self.train_dataset = Dataset.from_pandas(train_df)

    def format_question(self, sample, prompt=False):
        """Format the question for the DiffVQA task.

        Args:
            sample: The sample to format.
            prompt: Whether to add the answer to the prompt. Defaults to False.

        Returns:
            A conversation in the huggingface style and the images.
        """
        image_folder_path = os.path.join(
            self.mimic_path,
            "mimic-cxr-jpg",
            "2.0.0",
            "files",
            f"p{str(sample['subject_id'])[:2]}",
            f"p{sample['subject_id']}",
            f"s{sample['study_id']}",
        )
        list_files = os.listdir(image_folder_path)

        images = [
            Image.open(os.path.join(image_folder_path, imagePath))
            for imagePath in list_files
            if imagePath.endswith(".jpg")
        ]

        img_tokens = "<img>" * len(images)
        formatted_question = f"{img_tokens} {sample['question']}"
        formatted_answer = f"{sample['answer']}"
        if formatted_answer in ["yes", "no"]:
            formatted_question = (
                "Answer the following question with yes or no. " + formatted_question
            )

        question = [{"role": "user", "content": formatted_question}]
        if prompt:
            question.append({"role": "assistant", "content": formatted_answer})

        return (question, images)

    def get_correct_answer(self, sample):
        """Get the correct answer for the DiffVQA task.

        Args:
            sample: The sample to get the correct answer for.

        Returns:
            The correct answer.
        """
        return sample["answer"].lower().strip()

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(self.path, "DiffVQA", "mimic_pair_questions.csv")
        ):
            self.path = os.path.join(self.path, "DiffVQA")
            return

        os.makedirs(os.path.join(self.path, "DiffVQA"), exist_ok=True)

        username, password = self.engine.get_physionet_credentials()

        download_file(
            "https://physionet.org/files/medical-diff-vqa/1.0.0/mimic_pair_questions.csv?download",
            os.path.join(self.path, "DiffVQA", "mimic_pair_questions.csv"),
            username,
            password,
        )

        self.path = os.path.join(self.path, "DiffVQA")
