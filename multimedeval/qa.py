from datasets import load_dataset, Dataset
from multimedeval.utils import cleanStr
from torchmetrics.text import BLEUScore
from multimedeval.taskFamilies import QA
import pandas as pd


class MedQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedQA"
        self.modality = "General medicine"

    def setup(self):
        cacheDir = self.engine.getConfig()["MedQA_dir"]

        if cacheDir is None:
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset(
            "bigbio/med_qa", name="med_qa_en_source", split="test", cache_dir=cacheDir, trust_remote_code=True
        )

        self.trainDataset = load_dataset(
            "bigbio/med_qa", name="med_qa_en_source", split="train", cache_dir=cacheDir, trust_remote_code=True
        )

        self.bleuScorer = BLEUScore(n_gram=1)

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = sample["options"]

        formattedQuestion = f"{question}\n"
        formattedQuestion += (
            "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}.' for option in options]) + "\n"
        )
        formattedQuestion += "What is the correct answer?"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            formattedAnswer = "The answer is " + sample["answer_idx"] + "."
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        if fullText:
            return f"{sample['answer_idx']}: {sample['answer'].lower().strip()}"
        else:
            return sample["answer_idx"].lower().strip()

    def getPredictedAnswer(self, pred: str, sample):
        pred = cleanStr(pred)
        if len(pred) == 0:
            return "Invalid answer"

        options = [cleanStr(f'{option["key"]} {option["value"]}') for option in sample["options"]]
        # Compute the BLEU score for each option
        scores = [self.bleuScorer([pred], [[option]]) for option in options]

        if max(scores) == 0:
            return "Invalid answer"

        pred = sample["options"][scores.index(max(scores))]["key"].lower()

        return pred


class PubMedQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "PubMedQA"
        self.modality = "General medicine"
        self.task = "QA"

    def setup(self):
        cacheDir = self.engine.getConfig()["PubMedQA_dir"]

        if cacheDir is None:
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="test",
            cache_dir=cacheDir,
            trust_remote_code=True,
        )

        self.trainDataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="train",
            cache_dir=cacheDir,
            trust_remote_code=True,
        )

    def getCorrectAnswer(self, sample, fullText=False):
        return sample["answer"][0]

    def format_question(self, sample, prompt=False):
        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"]

        formattedQuestion = "Answer the question with yes, no or maybe. "
        formattedQuestion += f"{context} {question}"

        formattedAnswer = answer[0]

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return (question, [])

    def getPredictedAnswer(self, pred: str, sample):
        pred = cleanStr(pred)
        if len(pred) == 0:
            return "Invalid answer"

        optionVocabs = [["yes"], ["no"], ["maybe"]]

        pred = pred.split(" ")
        scores = [0 for _ in range(len(optionVocabs))]
        for token in pred:
            for idx in range(len(optionVocabs)):
                if token in optionVocabs[idx]:
                    scores[idx] += 1

        if max(scores) == 0:
            return "Invalid answer"

        pred = optionVocabs[scores.index(max(scores))][0].lower()
        return pred


class MedMCQA(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedMCQA"
        self.modality = "General medicine"
        self.task = "QA"

    def setup(self):
        cacheDir = self.engine.getConfig()["MedMCQA_dir"]

        if cacheDir is None:
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset("medmcqa", split="validation", cache_dir=cacheDir)

        self.trainDataset = load_dataset("medmcqa", split="train", cache_dir=cacheDir)

        self.bleuScorer = BLEUScore(n_gram=1)

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = self._getOptions(sample)
        answer = sample["cop"]

        formattedQuestion = f"{question}\n"
        formattedQuestion += "\n".join(options) + "\n"
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
            f"a: {sample['opa']}.",
            f"b: {sample['opb']}.",
            f"c: {sample['opc']}.",
            f"d: {sample['opd']}.",
        ]

    def getPredictedAnswer(self, pred: str, sample):
        pred = cleanStr(pred)
        if len(pred) == 0:
            return "Invalid answer"

        # Compute the BLEU score for each option
        scores = [self.bleuScorer([pred], [[cleanStr(option)]]) for option in self._getOptions(sample)]

        if max(scores) == 0:
            return "Invalid answer"

        pred = str(scores.index(max(scores)) + 1)  # +1 because the options are 1, 2, 3, 4 and not 0, 1, 2, 3
        return pred


class MMLU(QA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MMLU"
        self.modality = "General knowledge"
        self.task = "QA"

    def setup(self):
        cacheDir = self.engine.getConfig()["MMLU_dir"]

        if cacheDir is None:
            raise Exception("No path for MMLU dataset provided in the config file. Skipping the task.")

        subsets = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

        # self.dataset = []
        # for subset in subsets:
        #     currentSubset = load_dataset("cais/mmlu", subset, split="test", cache_dir=cacheDir)
        #     self.dataset += currentSubset.to_list()[: len(currentSubset) // 4]

        self.dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=cacheDir)
        self.dataset = self.dataset.to_pandas()

        def keep_quarter(group):
            quarter_len = len(group) // 4  # Calculate the length of a quarter
            return group.head(quarter_len)  # Keep the first quarter of rows
            
        self.dataset = self.dataset.groupby('subject').apply(keep_quarter)
        self.dataset = Dataset.from_pandas(self.dataset)

        self.trainDataset = load_dataset("cais/mmlu", "all", split="dev", cache_dir=cacheDir)

        self.bleuScorer = BLEUScore(n_gram=1)

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = self._getOptions(sample)
        answer = sample["answer"] # Answer is the index of the correct option

        formattedQuestion = f"{question}\n"
        formattedQuestion += "\n".join(options) + "\n"
        formattedQuestion += "Answer:"

        formattedAnswer = f"The answer is {options[answer]}."

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})
        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        number = sample["answer"]
        if fullText:
            return self._getOptions(sample)[number]
        else:
            return str(number + 1)

    def _getOptions(self, sample):
        choices = sample["choices"]
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        options = [f"{idx}. {choice}." for idx, choice in zip(letters, choices)]
        return options

    def getPredictedAnswer(self, pred: str, sample):
        pred = cleanStr(pred)
        if len(pred) == 0:
            return "Invalid answer"

        # Compute the BLEU score for each option
        scores = [self.bleuScorer([pred], [[cleanStr(option)]]) for option in self._getOptions(sample)]

        if max(scores) == 0:
            return "Invalid answer"

        pred = str(scores.index(max(scores)) + 1)

        return pred
