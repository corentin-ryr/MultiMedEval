from datasets import load_dataset
from multimedbench.utils import Benchmark, batchSampler, Params
from tqdm import tqdm
import math
from multimedbench.utils import remove_punctuation
import random

# from nltk.corpus import stopwords
# import nltk
from abc import abstractmethod
from torchmetrics import BLEUScore

# nltk.download("stopwords")
# STOPWORDS = stopwords.words("english")
# STOPWORDS.remove("a")
# STOPWORDS.remove("d")

import json


class QA(Benchmark):
    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        correct_answers = 0
        total_answers = 0

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
                if self.fewshot:
                    batchPrompts.append((self.prompt[0] + text, self.prompt[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                isCorrect = False
                if self.isValid(answer, batch[idx]):
                    correct_answers += 1
                    isCorrect = True
                total_answers += 1

                answersLog.append((self.getCorrectAnswer(batch[idx], fullText=True), answer, isCorrect))

        # TODO: add others metrics such as AUC, F1...
        metrics = {"accuracy": correct_answers / total_answers}

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

    @abstractmethod
    def format_question(self, sample, prompt=False):
        pass

    @abstractmethod
    def getCorrectAnswer(self, sample, fullText=False):
        pass

    @abstractmethod
    def isValid(self, pred: str, sample):
        pass


class MedQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedQA"

        params = json.load(open("MedMD_config.json", "r"))

        self.dataset = load_dataset(
            "bigbio/med_qa", name="med_qa_en_source", split="test", cache_dir=params["MedQA"]["path"]
        )

        self.trainDataset = load_dataset("bigbio/med_qa", split="train", cache_dir=params["MedQA"]["path"])

        self.prompt = self.getPrompt()

        self.bleuScorer = BLEUScore()

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = sample["options"]
        answer = f"{sample['answer_idx']}: {sample['answer']}"

        formattedQuestion = f"{question}\n"
        formattedQuestion += (
            "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}.' for option in options]) + "\n"
        )
        formattedQuestion += "What is the correct answer?"
        formattedAnswer = "The answer is " + (answer if prompt else "") + "."

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        if fullText:
            return f"{sample['answer_idx']}: {sample['answer'].lower().strip()}"
        else:
            return sample["answer_idx"].lower().strip()

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False

        options = [self.cleanStr(f'{option["key"]}: {option["value"]}') for option in sample["options"]]
        # Compute the BLEU score for each option
        scores = [self.bleuScorer([self.cleanStr(option)], [pred]) for option in options]

        pred = sample["options"][scores.index(max(scores))]["key"].lower()

        gold = self.getCorrectAnswer(sample)
        return pred == gold


class PubMedQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "PubMedQA"

        params = json.load(open("MedMD_config.json", "r"))

        self.dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="test",
            cache_dir=params["PubMedQA"]["path"],
        )

        self.trainDataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="train",
            cache_dir=params["PubMedQA"]["path"],
        )

        self.prompt = self.getPrompt()

    def getCorrectAnswer(self, sample, fullText=False):
        return sample["answer"]

    def format_question(self, sample, prompt=False):
        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"]

        formattedQuestion = f"{context}\nQuestion: {question}\n"
        formattedQuestion += "Answer with yes, no or maybe."

        formattedAnswer = answer[0]

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return (question, [])

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False

        optionVocabs = [["yes"], ["no"], ["maybe"]]

        pred = pred.split(" ")
        scores = [0 for _ in range(len(optionVocabs))]
        for token in pred:
            for idx in range(len(optionVocabs)):
                if token in optionVocabs[idx]:
                    scores[idx] += 1

        if max(scores) == 0:
            return False

        pred = optionVocabs[scores.index(max(scores))][0].lower()

        gold = self.getCorrectAnswer(sample)[0]
        return pred == gold


class MedMCQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedMCQA"

        params = json.load(open("MedMD_config.json", "r"))

        self.dataset = load_dataset(
            "medmcqa",
            split="test",
            cache_dir=params["MedMCQA"]["path"],
        )

        self.trainDataset = load_dataset(
            "medmcqa",
            split="train",
            cache_dir=params["MedMCQA"]["path"],
        )

        self.prompt = self.getPrompt()

        self.bleuScorer = BLEUScore()

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = [
            f"1: {sample['opa']}.",
            f"2: {sample['opb']}.",
            f"3: {sample['opc']}.",
            f"4: {sample['opd']}.",
        ]
        answer = sample["cop"]

        formattedQuestion = f"{question}\n"
        formattedQuestion += "Options:\n" + "\n".join(options) + "\n"
        formattedQuestion += "What is the correct answer?"

        formattedAnswer = f"The answer is {options[answer]}."

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        number = sample["cop"]
        if fullText:
            return self._getOptions(sample)[number]
        else:
            return str(number + 1)

    def _getOptions(self, sample):
        return [
            f"1: {sample['opa']}.",
            f"2: {sample['opb']}.",
            f"3: {sample['opc']}.",
            f"4: {sample['opd']}.",
        ]

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False

        options = [f"1: {sample['opa']}.", f"2: {sample['opb']}.", f"3: {sample['opc']}.", f"4: {sample['opd']}."]
        # Compute the BLEU score for each option
        scores = [self.bleuScorer([self.cleanStr(option)], [pred]) for option in options]

        pred = scores.index(max(scores)) + 1  # +1 because the options are 1, 2, 3, 4 and not 0, 1, 2, 3

        gold = self.getCorrectAnswer(sample)[0]
        return pred == gold
