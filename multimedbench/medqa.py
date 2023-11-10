from datasets import load_dataset
from multimedbench.utils import Benchmark, batchSampler, Params
from tqdm import tqdm 
import math
from multimedbench.utils import remove_punctuation
import random
import csv


class MedQA(Benchmark):
    def __init__(self, data_folder="/home/croyer/data", seed=1111):
        self.taskName = "MedQA"
        self.seed = seed
        random.seed(self.seed)

        # Get the dataset
        self.dataset = load_dataset(
            "bigbio/med_qa", split="test", cache_dir=f"{data_folder}/medqa"
        )

        # Create the prompt base
        self.trainDataset = load_dataset(
            "bigbio/med_qa", split="train", cache_dir=f"{data_folder}/medqa"
        )

        self.prompt = self.getPrompt()

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
            batchPrompts = [(self.format_question(sample), {}) for sample in batch]

            answers = batcher(batchPrompts, self.prompt)

            for idx, answer in enumerate(answers):
                if self.isValid(answer, self.getCorrectAnswer(batch[idx])):
                    correct_answers += 1
                total_answers += 1

                answersLog.append((self.getCorrectAnswer(batch[idx]), answer))

        # Write answersLog to a csv file
        try:
            with open(f"answerLog{self.taskName}.csv", "w", newline="") as f:
                spamWriter = csv.writer(f)
                spamWriter.writerows(answersLog)
        except Exception as e:
            print(e)

        # Compute the scores
        return {"accuracy": correct_answers / total_answers}

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer_idx"]

        formattedQuestion = f"Question:\n {question}\n"
        formattedQuestion += "Options:\n" + "\n".join(
            [f'{option["key"]}: {option["value"]}' for option in options]
        )
        formattedAnswer = "\nThe answer is " + (f"{answer}." if prompt else "")

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return question

    def getCorrectAnswer(self, sample):
        return sample["answer_idx"]

    def isValid(self, pred: str, gold: str):
        pred = remove_punctuation(
            pred.lower().replace("\n", "").replace("the answer is", "").strip()
        )
        if len(pred) == 0:
            return False
        gold = remove_punctuation(gold.lower().replace("\n", "").strip())[0]
        return pred[0] == gold
    
    def getPrompt(self):
        prompt = []
        for _ in range(3):
            prompt += self.format_question(
                self.trainDataset[random.randint(0, len(self.trainDataset))], prompt=True
            )
        return prompt


class PubMedQA(MedQA):
    def __init__(self, data_folder="/home/croyer/data", seed=42):
        self.taskName = "PubMedQA"
        self.seed = seed
        random.seed(self.seed)

        # Get the dataset
        self.dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="test",
            cache_dir=f"{data_folder}/pubmedqa",
        )
        # Prepare the prompt base
        self.trainDataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="train",
            cache_dir=f"{data_folder}/pubmedqa",
        )

        self.prompt = self.getPrompt()

    def getCorrectAnswer(self, sample):
        return sample["answer"][0]

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        answer = sample["answer"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options: yes, no or maybe"
        formattedAnswer = f"\nThe answer is {answer[0]}."

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return question


class MedMCQA(MedQA):
    def __init__(self, data_folder="/home/croyer/data", seed=1111):
        self.taskName = "MedMCQA"
        self.seed = seed
        random.seed(self.seed)

        # Get the dataset
        self.dataset = load_dataset(
            "medmcqa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="validation",
            cache_dir=f"{data_folder}/medmcqa",
        )

        # Prepare the prompt base
        self.trainDataset = load_dataset(
            "medmcqa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="train",
            cache_dir=f"{data_folder}/medmcqa",
        )
        print(self.trainDataset[0])

        self.prompt = self.getPrompt()

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = [
            f"A: {sample['opa']}",
            f"B: {sample['opb']}",
            f"C: {sample['opc']}",
            f"D: {sample['opd']}",
        ]
        answer = sample["cop"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options:\n" + "\n".join(options)
        formattedAnswer = f"\nThe answer is {chr(ord('a') + answer - 1).upper()}."

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return question

    def getCorrectAnswer(self, sample):
        return chr(ord("a") + sample["cop"] - 1).upper()

