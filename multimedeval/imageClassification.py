from multimedeval.utils import cleanStr
from multimedeval.tqdm_loggable import tqdm_logging
import os
import pandas as pd
import datasets
from PIL import Image
import pydicom
import numpy as np
import urllib.request
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from multimedeval.utils import download_file
from zipfile import ZipFile
import subprocess
from multimedeval.taskFamilies import ImageClassification


class MIMIC_CXR_ImageClassification(ImageClassification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.taskName = "MIMIC-CXR Image Classficication"
        self.modality = "X-Ray"

    def setup(self):
        self.scoringType = "multilabel"

        self.num_classes = 5
        self.path = self.engine.getConfig()["MIMIC_CXR_dir"]
        self._generate_dataset()

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        chexbertMimic = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-chexpert.csv"))

        testSplit = split[split.split == "test"]
        chexbertMimicTest = chexbertMimic[chexbertMimic.study_id.isin(testSplit.study_id)]
        chexbertMimicTest = chexbertMimicTest.merge(testSplit, on=["study_id", "subject_id"])
        self.dataset = datasets.Dataset.from_pandas(chexbertMimicTest)

        trainSplit = split[split.split == "train"]
        chexbertMimicTrain = chexbertMimic[chexbertMimic.study_id.isin(trainSplit.study_id)]
        chexbertMimicTrain = chexbertMimicTrain.merge(trainSplit, on=["study_id", "subject_id"])
        self.trainDataset = datasets.Dataset.from_pandas(chexbertMimicTrain)

        self.labelNames = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            "No Finding",
        ]
        self.conditions = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]

    def format_question(self, sample, prompt=False):
        samplePath = os.path.join(
            self.path, "files", "p" + str(sample["subject_id"])[:2], "p" + str(sample["subject_id"])
        )
        imagePath = os.path.join(samplePath, "s" + str(sample["study_id"]), sample["dicom_id"] + ".jpg")
        question = "<img> List the conditions that can be seen in this picture."
        formattedText = [
            {
                "role": "user",
                "content": question,
            }
        ]
        answer = self.getCorrectAnswer(sample, fullText=True)

        if prompt:
            formattedText.append({"role": "assistant", "content": answer})

        return (formattedText, [Image.open(imagePath)])

    def getPredictedAnswer(self, answer: str) -> int:
        df = pd.DataFrame(columns=["Report Impression"], data=[answer])
        labels = [element[0] == 1 for element in self.engine.labeler(df)]
        labels = [int(labels[self.labelNames.index(condition)]) for condition in self.conditions]

        return labels

    def getCorrectAnswer(self, sample, fullText=False) -> int:
        # Features: [Atelectasis, cardiomegaly, consolidation, edema, and pleural effusion]
        # If any of the features is 1, then it is positive
        # If all the features are 0, -1 or NaN, then it is negative
        labels = [int(sample[condition] == 1) for condition in self.conditions]

        if fullText:
            return ", ".join([self.conditions[idx] for idx, label in enumerate(labels) if label])

        return labels

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(os.path.join(self.path, "mimic-cxr-jpg", "2.0.0", "mimic-cxr-2.0.0-split.csv")):
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


class VinDr_Mammo(ImageClassification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.taskName = "VinDr Mammo"
        self.modality = "Mammography"

    def setup(self):

        self.path = self.engine.getConfig()["VinDr_Mammo_dir"]

        self._generate_dataset()

        self.num_classes = 5
        self.scoringType = "multiclass"
        self.options = ["1", "2", "3", "4", "5"]

        # Open the finding_annotation.csv file
        annotations = pd.read_csv(os.path.join(self.path, "finding_annotations.csv"))
        annotationsTest = annotations[annotations["split"] == "test"]

        # Only keep rows where "finding_birads" is not None
        annotationsTest = annotationsTest[annotationsTest["finding_birads"].notna()]

        self.dataset = datasets.Dataset.from_pandas(annotationsTest)

        annotationsTrain = annotations[annotations["split"] == "training"]
        annotationsTrain = annotationsTrain[annotationsTrain["finding_birads"].notna()]
        self.trainDataset = datasets.Dataset.from_pandas(annotationsTrain)

    def format_question(self, sample, prompt=False):
        formattedText = [
            {
                "role": "user",
                "content": f"<img> What is the BI-RADS level in this mammography (from 1 to 5)?",
            }
        ]

        if prompt:
            formattedText.append({"role": "assistant", "content": f"{sample['finding_birads']}"})

        dicom = pydicom.dcmread(os.path.join(self.path, "images", sample["study_id"], f"{sample['image_id']}.dicom"))

        dicom.BitsStored = 16
        data = dicom.pixel_array

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        image = Image.fromarray(data)
        return (formattedText, [image])

    def getPredictedAnswer(self, answer: str) -> int:
        # # Find the numbers in the string answer
        # findings = [int(s) for s in answer.split() if s.isdigit()]
        # if len(findings) > 0 and findings[0] <= 5 and findings[0] >= 1:
        #     return findings[0] - 1
        # else:
        #     return random.randint(0, 4)  # 5 classes so 0 to 4

        answer = cleanStr(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[cleanStr(option)]]) for option in self.options]
        return scores.index(max(scores))

    def getCorrectAnswer(self, sample, fullText=False) -> int:
        findings = sample["finding_birads"]
        findings = int(findings[-1])

        return findings - 1  # 5 classes so 0 to 4

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0", "finding_annotations.csv")
        ):
            self.path = os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0")
            return

        os.makedirs(self.path, exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        # wget_command = f'wget -c --user "{username}" --password "{password}" -O "{self.path}vindr_mammo.zip" https://physionet.org/content/vindr-mammo/get-zip/1.0.0/'
        # subprocess.run(wget_command, shell=True, check=True)

        download_file(
            "https://physionet.org/content/vindr-mammo/get-zip/1.0.0/",
            os.path.join(self.path, "vindr_mammo.zip"),
            username,
            password,
        )

        # Unzip the file
        with ZipFile(os.path.join(self.path, "vindr_mammo.zip"), "r") as zipObj:
            zipObj.extractall(self.path)

        self.path = os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0")


class Pad_UFES_20(ImageClassification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = "Pad UFES 20"
        self.modality = "Dermatology"

    def setup(self):
        self.num_classes = 7
        self.scoringType = "multiclass"

        self.path = self.engine.getConfig()["Pad_UFES_20_dir"]

        # Check if the folder contains the zip file
        if not os.path.exists(os.path.join(self.path, "pad_ufes_20.zip")):
            self._generate_dataset()

        dataset = pd.read_csv(os.path.join(self.path, "metadata.csv"))

        self.options = ["BCC", "SCC", "ACK", "SEK", "BOD", "MEL", "NEV"]
        self.mapAcronymToName = {
            "BCC": "Basal Cell Carcinoma (BCC)",
            "SCC": "Squamous Cell Carcinoma (SCC)",
            "ACK": "Actinic Keratosis (ACK)",
            "SEK": "Seborrheic Keratosis (SEK)",
            "BOD": "Bowen’s disease (BOD)",
            "MEL": "Melanoma (MEL)",
            "NEV": "Nevus (NEV)",
        }

        split = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "padufessplit.csv"))
        split = split[split["split"] == "test"]
        split = split["ids"].tolist()

        self.dataset = dataset[dataset["lesion_id"].isin(split)]
        self.dataset = datasets.Dataset.from_pandas(dataset)

        self.trainDataset = dataset[~dataset["lesion_id"].isin(split)]
        self.trainDataset = datasets.Dataset.from_pandas(self.trainDataset)


    def format_question(self, sample, prompt=False):
        patientInfo = {
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
        # patientInfo = ", ".join([f"{key} {value}" for key, value in patientInfo.items() if value is not None])
        options = "Options:\n" + "\n".join([self.mapAcronymToName[option] for option in self.options])

        formattedText = [
            {
                "role": "user",
                "content": f"<img> {options} What is the most likely diagnosis among the following propositions?",
            }
        ]

        if prompt:
            formattedText.append({"role": "assistant", "content": f"{self.mapAcronymToName[sample['diagnostic']]} ({sample['diagnostic']})"})

        image = Image.open(os.path.join(self.path, "images", sample["img_id"]))
        return (formattedText, [image])

    def getPredictedAnswer(self, answer: str) -> int:
        answer = cleanStr(answer)
        # Find the best bleu score between the answer and the options
        options = [cleanStr(self.mapAcronymToName[option]) for option in self.options]
        scores = [self.bleu([answer], [[option]]) for option in options]

        return scores.index(max(scores))

    def getCorrectAnswer(self, sample, fullText=False) -> int:
        correctName = sample["diagnostic"]

        if fullText:
            return self.mapAcronymToName[correctName]

        return self.options.index(correctName)

    def _generate_dataset(self):
        dataFolder = self.path
        # Download the file
        self.logger.info("Downloading the dataset...")
        url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
        with tqdm_logging(logger=self.logger, unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]) as t:
            os.makedirs(dataFolder, exist_ok=True)
            urllib.request.urlretrieve(
                url, os.path.join(dataFolder, "pad_ufes_20.zip"), reporthook=lambda x, y, z: t.update(y)
            )

        # Extract the file
        self.logger.info("Extracting the dataset...")
        with zipfile.ZipFile(os.path.join(dataFolder, "pad_ufes_20.zip"), "r") as zip_ref:
            zip_ref.extractall(f"{dataFolder}")

        self.logger.info("Extracting the images...")
        for file in os.listdir(os.path.join(dataFolder, "images")):
            if not file.endswith(".zip"):
                continue
            with zipfile.ZipFile(os.path.join(dataFolder, "images", file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataFolder, "images"))
                os.remove(os.path.join(dataFolder, "images", file))

        self.logger.info("Copying the images...")
        for file in os.listdir(os.path.join(dataFolder, "images")):
            if not os.path.isdir(os.path.join(dataFolder, "images", file)):
                continue
            for image in os.listdir(os.path.join(dataFolder, "images", file)):
                shutil.copyfile(
                    os.path.join(dataFolder, "images", file, image), os.path.join(dataFolder, "images", image)
                )
                os.remove(os.path.join(dataFolder, "images", file, image))

            os.rmdir(os.path.join(dataFolder, "images", file))



class CBIS_DDSM(ImageClassification):
    def __init__(self, abnormality: str, **kwargs):
        super().__init__(**kwargs)
        self.modality = "Mammography"
        self.abnormality = abnormality

    def setup(self):
        self.num_classes = 3
        self.scoringType = "multiclass"

        # Get the dataset from Kaggle
        self.path = self.engine.getConfig()["CBIS_DDSM_dir"]

        # Download the file at address https://huggingface.co/datasets/Reverb/CBIS-DDSM/resolve/main/CBIS-DDSM.7z?download=true
        self._generate_dataset()

        # Open the calc_case_description_test_set.csv file with pandas
        df_dicom = pd.read_csv(os.path.join(self.path, "csv", "dicom_info.csv"))

        cropped_images = df_dicom[df_dicom.SeriesDescription == "cropped images"].image_path
        full_mammo = df_dicom[df_dicom.SeriesDescription == "full mammogram images"].image_path
        roi_img = df_dicom[df_dicom.SeriesDescription == "ROI mask images"].image_path
        nan_img = df_dicom[df_dicom.SeriesDescription.isna()].image_path

        self.full_mammo_dict = dict()
        self.cropped_images_dict = dict()
        self.roi_img_dict = dict()
        self.nan_dict = dict()

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

        self.dataset = pd.read_csv(os.path.join(self.path, "csv", f"{self.abnormality}_case_description_test_set.csv"))
        self.dataset = self.dataset[["pathology", "cropped image file path"]]
        self._fix_image_path(self.dataset)
        self.dataset = datasets.Dataset.from_pandas(self.dataset)

        self.trainDataset = pd.read_csv(
            os.path.join(self.path, "csv", f"{self.abnormality}_case_description_train_set.csv")
        )
        self.trainDataset = self.trainDataset[["pathology", "cropped image file path"]]
        self._fix_image_path(self.trainDataset)
        self.trainDataset = datasets.Dataset.from_pandas(self.trainDataset)

    def getPredictedAnswer(self, answer: str) -> int:
        answer = cleanStr(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[cleanStr(option)]]) for option in self.options]
        return scores.index(max(scores))

    def getCorrectAnswer(self, sample, fullText=False) -> int:
        if fullText:
            return sample["pathology"]

        return self.options.index(sample["pathology"])

    def _generate_dataset(self):
        if os.path.exists(os.path.join(self.path, "csv")):
            return

        api = KaggleApi()
        api.authenticate()

        os.makedirs(self.path, exist_ok=True)

        api.dataset_download_files("awsaf49/cbis-ddsm-breast-cancer-image-dataset", path=self.path, unzip=True)

    def _fix_image_path(self, data: pd.DataFrame):
        """correct dicom paths to correct image paths"""
        for idx in range(len(data)):
            sample = data.iloc[idx]

            img_name = sample["cropped image file path"].split("/")[2]
            if img_name in self.cropped_images_dict:
                imagePath = self.cropped_images_dict[img_name]
            elif img_name in self.nan_dict:
                imagePath = self.nan_dict[img_name]

            data.iloc[idx, data.columns.get_loc("cropped image file path")] = imagePath


class CBIS_DDSM_Calcification(CBIS_DDSM):
    def __init__(self, **kwargs) -> None:
        super().__init__(abnormality="calc", **kwargs)
        self.taskName = "CBIS-DDSM Calcification"

    def format_question(self, sample, prompt=False):
        path = Path(sample["cropped image file path"])
        path = Path(self.path) / Path(*path.parts[1:])

        formattedText = [
            {
                "role": "user",
                "content": f"<img> Is the calcification benign, malignant or benign without callback?",
            }
        ]
        if prompt:
            formattedText.append({"role": "assistant", "content": f"{sample['pathology'].lower()}"})

        image = Image.open(os.path.join(self.path, "images", path))
        return (formattedText, [image])


class CBIS_DDSM_Mass(CBIS_DDSM):
    def __init__(self, **kwargs) -> None:
        super().__init__(abnormality="mass", **kwargs)
        self.taskName = "CBIS-DDSM Mass"

    def format_question(self, sample, prompt=False):
        path = Path(sample["cropped image file path"])
        path = Path(self.path) / Path(*path.parts[1:])

        formattedText = [
            {
                "role": "user",
                "content": f"<img> Is the mass benign, malignant or benign without callback?",
            }
        ]
        if prompt:
            formattedText.append({"role": "assistant", "content": f"{sample['pathology'].lower()}"})

        image = Image.open(os.path.join(self.path, "images", path))
        return (formattedText, [image])
