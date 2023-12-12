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



class VQA(Benchmark):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)

        self.f1 = []
    
    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        answersLog = []

        # Run the batcher for all data split in chunks
        for batch in tqdm(
            batchSampler(self.dataset, params.batch_size),
            total=math.ceil(len(self.dataset) / params.batch_size),
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                batchPrompts.append((self.prompt[0] + text, self.prompt[1] + img))

            answers = batcher(batchPrompts)

            correctAnswers = [[self.getCorrectAnswer(sample)] for sample in batch]

            for idx, answer in enumerate(answers):
                answersLog.append(
                    (self.getCorrectAnswer(batch[idx]), answer)
                )

                # Compute the number of tokens recalled in the answer
                tokens = set(self.cleanStr(answer).split(" "))
                correctTokens = set(self.cleanStr(self.getCorrectAnswer(batch[idx])).split(" "))
                precision = len(tokens.intersection(correctTokens)) / len(tokens)
                recall = len(tokens.intersection(correctTokens)) / len(correctTokens)
                self.f1.append(2 * (precision * recall) / (precision + recall + 1e-8))

            self.bleu.update(answers, correctAnswers)

                    
        # TODO: add others metrics such as AUC, F1...
        metrics = {"bleu": self.bleu.compute().item(), "F1": sum(self.f1) / len(self.f1)}

        # Compute the scores
        return [{"type":"json", "name": f"metrics_{self.taskName}", "value": metrics}, {"type":"csv", "name": self.taskName, "value": answersLog}]
    
    def cleanStr(self, text:str):
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
    
    @abstractmethod
    def format_question(self, sample, prompt=False):
        pass

    @abstractmethod
    def getCorrectAnswer(self, sample):
        pass 



class VQA_RAD(VQA):
    def __init__(self, seed=1111, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = "VQA-Rad"

        params = json.load(open("MedMD_config.json", "r"))
        
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", cache_dir=params["VQA-Rad"]["path"])
        self.trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train", cache_dir=params["VQA-Rad"]["path"])

        self.prompt = self.getPrompt()


    def format_question(self, sample, prompt=False):

        formattedQuestion = f"<img>, {sample['question']}"
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

        params = json.load(open("MedMD_config.json", "r"))
        self.dataset = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=params["VQA-Path"]["path"])
        self.trainDataset = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir=params["VQA-Path"]["path"])

        self.prompt = self.getPrompt()

    def format_question(self, sample, prompt=False):

        formattedQuestion = f"<img>, {sample['question']}"
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

        params = json.load(open("MedMD_config.json", "r"))

        self.path = params["SLAKE"]["path"]

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

        formattedQuestion = f"<img>, {sample['question']}"
        formattedAnswer = f"{sample['answer']}"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        images = [PIL.Image.open(os.path.join(self.path, "imgs", sample["img_name"]))]

        return (question, images)

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()