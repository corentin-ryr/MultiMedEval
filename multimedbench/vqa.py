from datasets import load_dataset
from torchmetrics.text import BLEUScore
from multimedbench.utils import Benchmark, batchSampler, Params
from tqdm import tqdm
import math
from multimedbench.utils import remove_punctuation
import random

from abc import abstractmethod
import json
import os
import PIL
import gdown
from zipfile import ZipFile


class VQA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "VQA"

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        answersLog = []
        bleuScores = []
        f1 = []

        # Run the batcher for all data split in chunks
        for batch in tqdm(
            batchSampler(self.dataset, params.batch_size),
            total=math.ceil(len(self.dataset) / params.batch_size),
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                if self.fewshot:
                    batchPrompts.append((self.prompt[0] + text, self.prompt[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                # Compute the number of tokens recalled in the answer
                tokens = set(self.cleanStr(answer).split(" "))
                correctTokens = set(self.cleanStr(self.getCorrectAnswer(batch[idx])).split(" "))
                precision = len(tokens.intersection(correctTokens)) / len(tokens)
                recall = len(tokens.intersection(correctTokens)) / len(correctTokens)
                currentF1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                f1.append(currentF1)

                currentBleu = self.bleu(
                    [self.cleanStr(answer)], [[self.cleanStr(self.getCorrectAnswer(batch[idx]))]]
                ).item()
                bleuScores.append(currentBleu)

                answersLog.append((self.getCorrectAnswer(batch[idx]), answer, currentF1, currentBleu))

        metrics = {"bleu": sum(bleuScores) / len(bleuScores), "F1": sum(f1) / len(f1)}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def cleanStr(self, text: str):
        return remove_punctuation(text.lower().replace("\n", " ").strip())

    def getPrompt(self):
        prompt = []
        images = []
        for _ in range(3):
            text, img = self.format_question(
                self.trainDataset[random.randint(0, len(self.trainDataset))],
                prompt=True,
            )
            prompt += text
            images += img
        return (prompt, images)

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def format_question(self, sample, prompt=False):
        pass

    @abstractmethod
    def getCorrectAnswer(self, sample):
        pass


class VQA_RAD(VQA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = "VQA-Rad"
        self.modality = "Radiology"

        cacheDir = json.load(open("MedMD_config.json", "r"))["huggingfaceCacheDir"]["path"]

        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", cache_dir=cacheDir)
        self.trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train", cache_dir=cacheDir)

        self.prompt = self.getPrompt()

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"

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

        cacheDir = json.load(open("MedMD_config.json", "r"))["huggingfaceCacheDir"]["path"]
        self.dataset = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=cacheDir)
        self.trainDataset = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir=cacheDir)

        self.prompt = self.getPrompt()

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"

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

        params = json.load(open("MedMD_config.json", "r"))

        self.path = params["SLAKE"]["path"]

        self._generate_dataset()

        self.path = os.path.join(self.path, "Slake1.0")

        with open(os.path.join(self.path, "train.json"), "r") as f:
            jsonFile = json.load(f)

        self.trainDataset = []
        for sample in jsonFile:
            if sample["q_lang"] == "en":
                self.trainDataset.append(sample)

        with open(os.path.join(self.path, "test.json"), "r") as f:
            jsonFile = json.load(f)

        self.dataset = []
        for sample in jsonFile:
            if sample["q_lang"] == "en":
                self.dataset.append(sample)

        self.prompt = self.getPrompt()

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"<img> {sample['question']}"
        formattedAnswer = f"{sample['answer']}"

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
            if os.path.exists(os.path.join(self.path, "Slake1.0", "train.json")):
                return

        output = os.path.join(self.path, "Slake.zip")
        if not os.path.exists(output):
            gdown.download(id="1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U", output=output, quiet=False)

        with ZipFile(output, "r") as zObject:
            zObject.extractall(path=self.path)

        os.remove(output)