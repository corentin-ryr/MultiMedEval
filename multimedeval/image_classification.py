"""Image classification."""

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Union
from zipfile import ZipFile

import datasets
import numpy as np
import pandas as pd
import pydicom
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from multimedeval.chexbert.constants import CONDITIONS
from multimedeval.task_families import ImageClassification
from multimedeval.tqdm_loggable import TqdmLogging
from multimedeval.utils import clean_str, download_file


class MIMICCXRImageClassification(ImageClassification):
    """MIMIC-CXR Image Classification task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the MIMIC-CXR Image Classification task."""
        super().__init__(**kwargs)

        self.task_name = "MIMIC-CXR Image Classification"
        self.modality = "X-Ray"

    def setup(self):
        """Setup the MIMIC-CXR Image Classification task."""
        self.scoring_type = "multilabel"

        self.num_classes = 5
        self.path = self.engine.get_config()["mimic_cxr_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping MIMIC-CXR Image classification because the cache directory is not set."
            )

        self._generate_dataset()

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        chexbert_mimic = pd.read_csv(
            os.path.join(self.path, "mimic-cxr-2.0.0-chexpert.csv")
        )

        test_split = split[split.split == "test"]
        chexbert_mimic_test = chexbert_mimic[
            chexbert_mimic.study_id.isin(test_split.study_id)
        ]
        chexbert_mimic_test = chexbert_mimic_test.merge(
            test_split, on=["study_id", "subject_id"]
        )
        self.dataset = datasets.Dataset.from_pandas(chexbert_mimic_test)

        train_split = split[split.split == "train"]
        chexbert_mimic_train = chexbert_mimic[
            chexbert_mimic.study_id.isin(train_split.study_id)
        ]
        chexbert_mimic_train = chexbert_mimic_train.merge(
            train_split, on=["study_id", "subject_id"]
        )
        self.train_dataset = datasets.Dataset.from_pandas(chexbert_mimic_train)

        self.conditions = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]

    def format_question(self, sample, prompt=False):
        """Format the question for the MIMIC-CXR Image Classification task.

        Args:
            sample: The sample to format.
            prompt: Add the answer to the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        sample_path = os.path.join(
            self.path,
            "files",
            "p" + str(sample["subject_id"])[:2],
            "p" + str(sample["subject_id"]),
        )
        image_path = os.path.join(
            sample_path, "s" + str(sample["study_id"]), sample["dicom_id"] + ".jpg"
        )
        question = "<img> List the conditions that can be seen in this picture."
        formatted_text = [
            {
                "role": "user",
                "content": question,
            }
        ]
        answer = self.get_correct_answer(sample, full_text=True)

        if prompt:
            formatted_text.append({"role": "assistant", "content": answer})

        return (formatted_text, [Image.open(image_path)])

    def get_predicted_answer(self, answer: str) -> Union[int, List[int]]:
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            The labels of the answer.
        """
        df = pd.DataFrame(columns=["Report Impression"], data=[answer])
        labels = [element[0] == 1 for element in self.engine.labeler(df)]
        labels = [
            int(labels[CONDITIONS.index(condition)]) for condition in self.conditions
        ]

        return labels

    def get_correct_answer(self, sample, full_text=False) -> Union[int, str, List[int]]:
        """Get the correct answer labels.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Whether or not to return the raw text. Defaults to False.

        Returns:
            : The correct answer labels.
        """
        # Features: [Atelectasis, cardiomegaly, consolidation, edema, and pleural effusion]
        # If any of the features is 1, then it is positive
        # If all the features are 0, -1 or NaN, then it is negative
        labels = [int(sample[condition] == 1) for condition in self.conditions]

        if full_text:
            return ", ".join(
                [self.conditions[idx] for idx, label in enumerate(labels) if label]
            )

        return labels

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
        wget_command = f'wget -r -c -np -nc --directory-prefix "{self.path}" \
            --user "{username}" \
            --password "{password}" https://physionet.org/files/mimic-cxr-jpg/2.0.0/'

        subprocess.run(wget_command, shell=True, check=True)

        self.path = os.path.join(self.path, "mimic-cxr-jpg", "2.0.0")

        # Unzip the mimic-cxr-2.0.0-split file
        file = os.path.join(self.path, "mimic-cxr-2.0.0-split.csv")
        with ZipFile(file + ".gz", "r") as zip_obj:
            zip_obj.extractall(file)


class VinDrMammo(ImageClassification):
    """VinDr Mammo Image Classification task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the VinDr Mammo Image Classification task."""
        super().__init__(**kwargs)

        self.task_name = "VinDr Mammo"
        self.modality = "Mammography"

    def setup(self):
        """Setup the VinDr Mammo Image Classification task."""
        self.path = self.engine.get_config()["vindr_mammo_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping VinDr Mammo because the cache directory is not set."
            )

        self._generate_dataset()

        self.num_classes = 5
        self.scoring_type = "multiclass"
        self.options = ["1", "2", "3", "4", "5"]

        # Open the finding_annotation.csv file
        annotations = pd.read_csv(os.path.join(self.path, "finding_annotations.csv"))
        annotations_test = annotations[annotations["split"] == "test"]

        # Only keep rows where "finding_birads" is not None
        annotations_test = annotations_test[annotations_test["finding_birads"].notna()]

        self.dataset = datasets.Dataset.from_pandas(annotations_test)

        annotations_train = annotations[annotations["split"] == "training"]
        annotations_train = annotations_train[
            annotations_train["finding_birads"].notna()
        ]
        self.train_dataset = datasets.Dataset.from_pandas(annotations_train)

    def format_question(self, sample, prompt=False):
        """Format the question for the VinDr Mammo Image Classification task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer to the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        formatted_text = [
            {
                "role": "user",
                "content": "<img> What is the BI-RADS level in this mammography (from 1 to 5)?",
            }
        ]

        if prompt:
            formatted_text.append(
                {"role": "assistant", "content": f"{sample['finding_birads']}"}
            )

        dicom = pydicom.dcmread(
            os.path.join(
                self.path, "images", sample["study_id"], f"{sample['image_id']}.dicom"
            )
        )

        dicom.BitsStored = 16
        data = dicom.pixel_array

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        image = Image.fromarray(data)
        return (formatted_text, [image])

    def get_predicted_answer(self, answer: str) -> int:
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            The index of the answer.
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[clean_str(option)]]) for option in self.options]
        return scores.index(max(scores))

    def get_correct_answer(self, sample, full_text=False) -> int:
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Not used. Defaults to False.

        Returns:
            The correct answer.
        """
        findings = sample["finding_birads"]
        findings = int(findings[-1])

        return findings - 1  # 5 classes so 0 to 4

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(
                self.path,
                "physionet.org",
                "files",
                "vindr-mammo",
                "1.0.0",
                "finding_annotations.csv",
            )
        ):
            self.path = os.path.join(
                self.path, "physionet.org", "files", "vindr-mammo", "1.0.0"
            )
            return

        os.makedirs(self.path, exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()

        download_file(
            "https://physionet.org/content/vindr-mammo/get-zip/1.0.0/",
            os.path.join(self.path, "vindr_mammo.zip"),
            username,
            password,
        )

        # Unzip the file
        with ZipFile(os.path.join(self.path, "vindr_mammo.zip"), "r") as zip_obj:
            zip_obj.extractall(self.path)

        self.path = os.path.join(
            self.path, "physionet.org", "files", "vindr-mammo", "1.0.0"
        )


class PadUFES20(ImageClassification):
    """Pad-UFES-20 Image Classification task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the Pad-UFES-20 Image Classification task."""
        super().__init__(**kwargs)
        self.task_name = "Pad UFES 20"
        self.modality = "Dermatology"

    def setup(self):
        """Setup the Pad-UFES-20 Image Classification task."""
        self.num_classes = 7
        self.scoring_type = "multiclass"

        self.path = self.engine.get_config()["pad_ufes_20_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping Pad-UFES 20 because the cache directory is not set."
            )

        # Check if the folder contains the zip file
        if not os.path.exists(os.path.join(self.path, "pad_ufes_20.zip")):
            self._generate_dataset()

        dataset = pd.read_csv(os.path.join(self.path, "metadata.csv"))

        self.options = ["BCC", "SCC", "ACK", "SEK", "BOD", "MEL", "NEV"]
        self.map_acronym_to_name = {
            "BCC": "Basal Cell Carcinoma (BCC)",
            "SCC": "Squamous Cell Carcinoma (SCC)",
            "ACK": "Actinic Keratosis (ACK)",
            "SEK": "Seborrheic Keratosis (SEK)",
            "BOD": "Bowen’s disease (BOD)",
            "MEL": "Melanoma (MEL)",
            "NEV": "Nevus (NEV)",
        }

        split_dset = load_dataset(
            "croyer/Pad-UFES-20-split", cache_dir=self.path, split="test"
        )
        split = set(split_dset["ids"])

        self.dataset = dataset[dataset["lesion_id"].isin(split)]
        self.dataset = datasets.Dataset.from_pandas(dataset)

        self.train_dataset = dataset[~dataset["lesion_id"].isin(split)]
        self.train_dataset = datasets.Dataset.from_pandas(self.train_dataset)

    def format_question(self, sample, prompt=False):
        """Format the question for the Pad-UFES-20 Image Classification task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer to the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        patient_info = {
            "smokes": sample["smoke"],
            "drink": sample["drink"],
            "age": sample["age"],
            "backgroundFather": sample["background_father"],
            "backgroundMother": sample["background_mother"],
            "pesticide": sample["pesticide"],
            "gender": sample["gender"],
            "skin cancer history": sample["skin_cancer_history"],
            "cancer history": sample["cancer_history"],
            "has piped water": sample["has_piped_water"],
            "has sewage system": sample["has_sewage_system"],
        }
        # Create a sentence out of the patient information don't include Nones
        patient_info = "Patient history: " + ", ".join(
            [
                f"{key} {value}"
                for key, value in patient_info.items()
                if value is not None
            ]
        )
        # options = "Options:\n" + "\n".join(
        #     [self.mapAcronymToName[option] for option in self.options]
        # )

        formatted_text = [
            {
                "role": "user",
                "content": f"<img> {patient_info} Which of the following is "
                "the most likely diagnosis of the patient's skin lesion? {options}",
            }
        ]

        if prompt:
            formatted_text.append(
                {
                    "role": "assistant",
                    "content": f"{self.map_acronym_to_name[sample['diagnostic']]}"
                    f" ({sample['diagnostic']})",
                }
            )

        image = Image.open(os.path.join(self.path, "images", sample["img_id"]))
        return (formatted_text, [image])

    def get_predicted_answer(self, answer: str) -> int:
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            The index of the answer.
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        options = [
            clean_str(self.map_acronym_to_name[option]) for option in self.options
        ]
        scores = [self.bleu([answer], [[option]]) for option in options]

        return scores.index(max(scores))

    def get_correct_answer(self, sample, full_text=False) -> int:
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Returns the raw answer. Defaults to False.

        Returns:
            The correct answer.
        """
        correct_name = sample["diagnostic"]

        if full_text:
            return self.map_acronym_to_name[correct_name]

        return self.options.index(correct_name)

    def _generate_dataset(self):
        data_folder = self.path
        # Download the file
        self.logger.info("Downloading the dataset...")
        url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
        with TqdmLogging(
            logger=self.logger,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=url.split("/")[-1],
        ) as t:
            os.makedirs(data_folder, exist_ok=True)
            urllib.request.urlretrieve(
                url,
                os.path.join(data_folder, "pad_ufes_20.zip"),
                reporthook=lambda x, y, z: t.update(y),
            )

        # Extract the file
        self.logger.info("Extracting the dataset...")
        with zipfile.ZipFile(
            os.path.join(data_folder, "pad_ufes_20.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(f"{data_folder}")

        self.logger.info("Extracting the images...")
        for file in os.listdir(os.path.join(data_folder, "images")):
            if not file.endswith(".zip"):
                continue
            with zipfile.ZipFile(
                os.path.join(data_folder, "images", file), "r"
            ) as zip_ref:
                zip_ref.extractall(os.path.join(data_folder, "images"))
                os.remove(os.path.join(data_folder, "images", file))

        self.logger.info("Copying the images...")
        for file in os.listdir(os.path.join(data_folder, "images")):
            if not os.path.isdir(os.path.join(data_folder, "images", file)):
                continue
            for image in os.listdir(os.path.join(data_folder, "images", file)):
                shutil.copyfile(
                    os.path.join(data_folder, "images", file, image),
                    os.path.join(data_folder, "images", image),
                )
                os.remove(os.path.join(data_folder, "images", file, image))

            os.rmdir(os.path.join(data_folder, "images", file))


class CBISDDSM(ImageClassification):
    """CBIS-DDSM Image Classification task."""

    def __init__(self, abnormality: str, **kwargs):
        """Initialize the CBIS-DDSM Image Classification task."""
        super().__init__(**kwargs)
        self.modality = "Mammography"
        self.abnormality = abnormality

    def setup(self):
        """Setup the CBIS-DDSM Image Classification task."""
        self.num_classes = 3
        self.scoring_type = "multiclass"

        # Get the dataset from Kaggle
        self.path = self.engine.get_config()["cbis_ddsm_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping CBIS-DDSM because the cache directory is not set."
            )

        self._generate_dataset()

        # Open the calc_case_description_test_set.csv file with pandas
        df_dicom = pd.read_csv(os.path.join(self.path, "csv", "dicom_info.csv"))

        cropped_images = df_dicom[
            df_dicom.SeriesDescription == "cropped images"
        ].image_path
        full_mammo = df_dicom[
            df_dicom.SeriesDescription == "full mammogram images"
        ].image_path
        roi_img = df_dicom[df_dicom.SeriesDescription == "ROI mask images"].image_path
        nan_img = df_dicom[df_dicom.SeriesDescription.isna()].image_path

        self.full_mammo_dict = {}
        self.cropped_images_dict = {}
        self.roi_img_dict = {}
        self.nan_dict = {}

        for dicom in full_mammo:
            key = dicom.split("/")[2]
            self.full_mammo_dict[key] = dicom
        for dicom in cropped_images:
            key = dicom.split("/")[2]
            self.cropped_images_dict[key] = dicom
        for dicom in roi_img:
            key = dicom.split("/")[2]
            self.roi_img_dict[key] = dicom
        for dicom in nan_img:
            key = dicom.split("/")[2]
            self.nan_dict[key] = dicom

        self.options = ["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK"]

        self.dataset = pd.read_csv(
            os.path.join(
                self.path, "csv", f"{self.abnormality}_case_description_test_set.csv"
            )
        )
        self.dataset = self.dataset[["pathology", "cropped image file path"]]
        self._fix_image_path(self.dataset)
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

        self.train_dataset = pd.read_csv(
            os.path.join(
                self.path, "csv", f"{self.abnormality}_case_description_train_set.csv"
            )
        )
        self.train_dataset = self.train_dataset[
            ["pathology", "cropped image file path"]
        ]
        self._fix_image_path(self.train_dataset)
        self.train_dataset = datasets.Dataset.from_pandas(self.train_dataset)

    def get_predicted_answer(self, answer: str) -> int:
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            The index of the answer.
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[clean_str(option)]]) for option in self.options]
        return scores.index(max(scores))

    def get_correct_answer(self, sample, full_text=False) -> int:
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Returns the raw answer. Defaults to False.

        Returns:
            The correct answer.
        """
        if full_text:
            return sample["pathology"]

        return self.options.index(sample["pathology"])

    def _generate_dataset(self):
        if os.path.exists(os.path.join(self.path, "csv")):
            return

        api = KaggleApi()
        api.authenticate()

        os.makedirs(self.path, exist_ok=True)

        api.dataset_download_files(
            "awsaf49/cbis-ddsm-breast-cancer-image-dataset", path=self.path, unzip=True
        )

    def _fix_image_path(self, data: pd.DataFrame):
        """Correct dicom paths to correct image paths."""
        for idx in range(len(data)):
            sample = data.iloc[idx]

            img_name = sample["cropped image file path"].split("/")[2]
            if img_name in self.cropped_images_dict:
                image_path = self.cropped_images_dict[img_name]
            elif img_name in self.nan_dict:
                image_path = self.nan_dict[img_name]

            data.iloc[idx, data.columns.get_loc("cropped image file path")] = image_path


class CBISDDSMCalcification(CBISDDSM):
    """CBIS-DDSM Calcification Image Classification task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the CBIS-DDSM Calcification Image Classification task."""
        super().__init__(abnormality="calc", **kwargs)
        self.task_name = "CBIS-DDSM Calcification"

    def format_question(self, sample, prompt=False):
        """Format the question for the CBIS-DDSM Calcification Image Classification task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answe rto the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        path = Path(sample["cropped image file path"])
        path = Path(self.path) / Path(*path.parts[1:])

        formatted_text = [
            {
                "role": "user",
                "content": "<img> Is the calcification benign, "
                "malignant or benign without callback?",
            }
        ]
        if prompt:
            formatted_text.append(
                {"role": "assistant", "content": f"{sample['pathology'].lower()}"}
            )

        image = Image.open(os.path.join(self.path, "images", path))
        return (formatted_text, [image])


class CBISDDSMMass(CBISDDSM):
    """CBIS-DDSM Mass Image Classification task."""

    def __init__(self, **kwargs) -> None:
        """Initialize the CBIS-DDSM Mass Image Classification task."""
        super().__init__(abnormality="mass", **kwargs)
        self.task_name = "CBIS-DDSM Mass"

    def format_question(self, sample, prompt=False):
        """Format the question for the CBIS-DDSM Mass Image Classification task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer to the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        path = Path(sample["cropped image file path"])
        path = Path(self.path) / Path(*path.parts[1:])

        formatted_text = [
            {
                "role": "user",
                "content": "<img> Is the mass benign, malignant or benign without callback?",
            }
        ]
        if prompt:
            formatted_text.append(
                {"role": "assistant", "content": f"{sample['pathology'].lower()}"}
            )

        image = Image.open(os.path.join(self.path, "images", path))
        return (formatted_text, [image])
