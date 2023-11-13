from datasets import load_dataset
from multimedbench.utils import Benchmark, batchSampler, Params
from tqdm import tqdm
import math
from multimedbench.utils import remove_punctuation
import random
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
STOPWORDS.remove("a")
STOPWORDS.remove("d")


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
                isCorrect = False
                if self.isValid(answer, batch[idx]):
                    correct_answers += 1
                    isCorrect = True
                total_answers += 1

                answersLog.append(
                    (self.getCorrectAnswer(batch[idx]), answer, isCorrect)
                )
            
        # TODO: add others metrics such as AUC, F1...
        metrics = {"accuracy": correct_answers / total_answers}

        # Compute the scores
        return [{"type":"json", "name": f"metrics_{self.taskName}", "value": metrics}, {"type":"csv", "name": self.taskName, "value": answersLog}]
    
    def cleanStr(self, text:str):
        return remove_punctuation(text.lower().replace("\n", " ").strip())

    # Special functions that should be implemented by subclasses ================

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
        return sample["answer_idx"].lower().strip()

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False
        
        optionVocabs = [self.cleanStr(f'{option["key"]}: {option["value"]}').split(" ") for option in sample["options"]]
        optionVocabs = [[word for word in option if word not in STOPWORDS] for option in optionVocabs]

        pred = pred.split(" ")
        pred = [word for word in pred if word not in STOPWORDS]
        scores = [0 for _ in range(len(optionVocabs))]
        for token in pred:
            for idx in range(len(optionVocabs)):
                if token in optionVocabs[idx]:
                    scores[idx] += 1
        
        
        pred = sample["options"][scores.index(max(scores))]["key"].lower()

        gold = self.getCorrectAnswer(sample)
        return pred == gold

    def getPrompt(self):
        prompt = []
        for _ in range(3):
            prompt += self.format_question(
                self.trainDataset[random.randint(0, len(self.trainDataset))],
                prompt=True,
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
        return sample["answer"]

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
    
    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False
        
        optionVocabs = ["yes", "no", "maybe"]

        pred = pred.split(" ")
        scores = [0 for _ in range(len(optionVocabs))]
        for token in pred:
            for idx in range(len(optionVocabs)):
                if token in optionVocabs[idx]:
                    scores[idx] += 1
        
        
        pred = optionVocabs[scores.index(max(scores))].lower()

        gold = self.getCorrectAnswer(sample)[0]
        return pred == gold


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

        self.prompt = self.getPrompt()

        self.mapToLetters = ["a", "b", "c", "d"]

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
        return self.mapToLetters[sample["cop"]]
    
    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0:
            return False
        
        optionVocabs = [
            f"A: {sample['opa']}",
            f"B: {sample['opb']}",
            f"C: {sample['opc']}",
            f"D: {sample['opd']}",
        ]
        optionVocabs = [self.cleanStr(option).split(" ") for option in optionVocabs]


        pred = pred.split(" ")
        pred = [word for word in pred if word not in STOPWORDS]
        scores = [0 for _ in range(len(optionVocabs))]
        for token in pred:
            for idx in range(len(optionVocabs)):
                if token in optionVocabs[idx]:
                    scores[idx] += 1
        
        pred = self.mapToLetters[scores.index(max(scores))]

        gold = self.getCorrectAnswer(sample)[0]
        return pred == gold
