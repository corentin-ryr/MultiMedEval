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
from nltk.stem import WordNetLemmatizer
import re


class VQA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "VQA"
        self.wnl = WordNetLemmatizer()

    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        answersLog = [["correct", "predicted", "F1", "BLEU", "recall", "correct tokens", "predicted tokens"]]
        bleuScores = []
        f1 = []
        recall = []

        closedQuestions = []
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
                cleanCorrect = self.cleanStr(self.getCorrectAnswer(batch[idx]))
                cleanPredicted = self.cleanStr(answer)

                predictedTokens = set([self.wnl.lemmatize(token) for token in cleanPredicted.split(" ")])
                correctTokens = set([self.wnl.lemmatize(token) for token in cleanCorrect.split(" ")])
                # correctTokens.add("not")
                currentPrecision = len(predictedTokens.intersection(correctTokens)) / len(predictedTokens)
                currentRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens)
                currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
                f1.append(currentF1)
                recall.append(currentRecall)

                currentBleu = self.bleu([cleanPredicted], [[cleanCorrect]]).item()
                bleuScores.append(currentBleu)

                # If the question is closed, decide if it is correct or not
                if cleanCorrect in ["yes", "no"]:
                    closedQuestions.append(True)
                else:
                    closedQuestions.append(False)

                answersLog.append(
                    (
                        self.getCorrectAnswer(batch[idx]),
                        answer,
                        currentF1,
                        currentBleu,
                        currentRecall,
                        correctTokens,
                        predictedTokens,
                    )
                )

        # Compute the accuracy for closed questions
        closedQuestionsCorrect = 0
        openQuestionsRecall = []
        for idx, isClosed in enumerate(closedQuestions):
            if isClosed and recall[idx] >= 0.4:
                closedQuestionsCorrect += 1
            elif not isClosed:
                openQuestionsRecall.append(recall[idx])

        metrics = {
            "bleu": sum(bleuScores) / len(bleuScores),
            "F1": sum(f1) / len(f1),
            "recall": sum(recall) / len(recall),
            "closedQuestionsAccuracy": closedQuestionsCorrect / sum(closedQuestions),
            "openQuestionsRecall": sum(openQuestionsRecall) / len(openQuestionsRecall),
        }

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def cleanStr(self, text: str):
        tempStr = remove_punctuation(text.lower().replace("\n", " ").strip())
        return re.sub(" +", " ", tempStr)

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

        cacheDir = self.engine.getConfig()["huggingfaceCacheDir"]["path"]

        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", cache_dir=cacheDir)
        if self.engine.params.fewshot:
            self.trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train", cache_dir=cacheDir)
            self.prompt = self.getPrompt()

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

        cacheDir = self.engine.getConfig()["huggingfaceCacheDir"]["path"]
        self.dataset = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=cacheDir)
        if self.engine.params.fewshot:
            self.trainDataset = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir=cacheDir)
            self.prompt = self.getPrompt()

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

        self.path = self.engine.getConfig()["SLAKE"]["path"]

        self._generate_dataset()

        self.path = os.path.join(self.path, "Slake1.0")

        jsonFile = json.load(open(os.path.join(self.path, "test.json"), "r"))

        self.dataset = []
        for sample in jsonFile:
            if sample["q_lang"] == "en":
                self.dataset.append(sample)

        if self.engine.params.fewshot:
            jsonFile = json.load(open(os.path.join(self.path, "train.json"), "r"))
            self.trainDataset = []
            for sample in jsonFile:
                if sample["q_lang"] == "en":
                    self.trainDataset.append(sample)
            self.prompt = self.getPrompt()

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
            if os.path.exists(os.path.join(self.path, "Slake1.0", "train.json")):
                return

        output = os.path.join(self.path, "Slake.zip")
        if not os.path.exists(output):
            gdown.download(id="1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U", output=output, quiet=False)

        with ZipFile(output, "r") as zObject:
            zObject.extractall(path=self.path)

        os.remove(output)
