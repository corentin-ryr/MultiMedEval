from datasets import load_dataset
import json
import os
import PIL
import gdown
from zipfile import ZipFile
from multimedeval.taskFamilies import VQA



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
        self._generate_dataset()


        if self.path is None:
            raise Exception("No path for SLAKE dataset provided in the config file. Skipping the task.")
        
        dset = load_dataset("BoKelvin/SLAKE", cache_dir=self.path)


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



    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"
        if formattedAnswer in ["yes", "no"]:
            formattedQuestion = "Answer the following question with yes or no. " + formattedQuestion

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        images = [PIL.Image.open(os.path.join(self.path, "imgs", sample["img_name"]))]

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
