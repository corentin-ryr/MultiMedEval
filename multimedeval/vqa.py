from datasets import load_dataset, Dataset
import os
from PIL import Image
import gdown
from zipfile import ZipFile
from multimedeval.taskFamilies import VQA
from multimedeval.utils import download_file
import pandas as pd




class VQA_RAD(VQA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = "VQA-Rad"
        self.modality = "Radiology"

    def setup(self):
        cacheDir = self.engine.getConfig()["VQA_RAD_dir"]

        if cacheDir is None:
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", cache_dir=cacheDir)
        self.trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train", cache_dir=cacheDir)

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"
        if formattedAnswer in ["yes", "no"]:
            formattedQuestion = "Answer the following question with yes or no. " + formattedQuestion

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [sample["image"]])

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()


class Path_VQA(VQA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "VQA-Path"
        self.modality = "Pathology"

    def setup(self):
        cacheDir = self.engine.getConfig()["Path_VQA_dir"]

        if cacheDir is None:
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")
    
        self.dataset = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=cacheDir)
        self.trainDataset = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir=cacheDir)

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"
        if formattedAnswer in ["yes", "no"]:
            formattedQuestion = "Answer the following question with yes or no. " + formattedQuestion

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [sample["image"]])

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()


class SLAKE(VQA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "SLAKE"
        self.modality = "Radiology"


    def setup(self):
        self.path = self.engine.getConfig()["SLAKE_dir"]

        if self.path is None:
            raise Exception("No path for SLAKE dataset provided in the config file. Skipping the task.")


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
        self.trainDataset = []
        for sample in dset["train"]:
            if sample["q_lang"] == "en":
                sample["image"] = os.path.join(self.path, "imgs", sample["img_name"])
                self.trainDataset.append(sample)

        self.dataset = Dataset.from_list(self.dataset)
        self.trainDataset = Dataset.from_list(self.trainDataset)



    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"
        if formattedAnswer in ["yes", "no"]:
            formattedQuestion = "Answer the following question with yes or no. " + formattedQuestion

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        images = [Image.open(os.path.join(self.path, "imgs", sample["img_name"]))]

        return (question, images)

    def getCorrectAnswer(self, sample):
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
            gdown.download("https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip?download=true", output=output, quiet=False)

        print(output)
        with ZipFile(output, "r") as zObject:
            zObject.extractall(path=os.path.join(self.path, "BoKelvin___slake"))

        os.remove(output)



class DiffVQA(VQA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "Diff-VQA"
        self.modality = "Radiology"


    def setup(self):
        self.path = self.engine.getConfig()["DiffVQA_dir"]

        self.mimicPath = self.engine.getConfig()["MIMIC_CXR_dir"]

        if self.path is None or self.mimicPath is None:
            raise Exception("No path for DiffVQA dataset provided in the config file. Skipping the task.")
    
        self._generate_dataset()

        # Open the csv file
        df = pd.read_csv(os.path.join(self.path, "mimic_pair_questions.csv"))

        testDf = df[df["split"] == "test"]
        self.dataset = Dataset.from_pandas(testDf)

        trainDf = df[df["split"] == "train"]
        self.trainDataset = Dataset.from_pandas(trainDf)



    def format_question(self, sample, prompt=False):
        imageFolderPath = os.path.join(self.mimicPath, "mimic-cxr-jpg", "2.0.0", "files", f"p{str(sample['subject_id'])[:2]}", f"p{sample['subject_id']}", f"s{sample['study_id']}")
        listOfFiles = os.listdir(imageFolderPath)

        images = [Image.open(os.path.join(imageFolderPath, imagePath)) for imagePath in listOfFiles if imagePath.endswith(".jpg")]

        imgTokens = "<img>" * len(images)
        formattedQuestion = f"{imgTokens} {sample['question']}"
        formattedAnswer = f"{sample['answer']}"
        if formattedAnswer in ["yes", "no"]:
            formattedQuestion = "Answer the following question with yes or no. " + formattedQuestion

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, images)

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(os.path.join(self.path, "DiffVQA", "mimic_pair_questions.csv")):
            self.path = os.path.join(self.path, "DiffVQA")
            return

        os.makedirs(os.path.join(self.path, "DiffVQA"), exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        # wget_command = f'wget -r -N -c -np --directory-prefix "{self.path}" --user "{username}" --password "{password}" https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz'
        # subprocess.run(wget_command, shell=True, check=True)

        download_file(
            "https://physionet.org/files/medical-diff-vqa/1.0.0/mimic_pair_questions.csv?download",
            os.path.join(self.path, "DiffVQA", "mimic_pair_questions.csv"),
            username,
            password,
        )

        self.path = os.path.join(self.path, "DiffVQA")

