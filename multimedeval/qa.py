from datasets import load_dataset
from multimedeval.utils import cleanStr
from torchmetrics.text import BLEUScore
from multimedeval.taskFamilies import QA


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
            raise Exception("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=cacheDir)

        self.trainDataset = load_dataset("cais/mmlu", "all", split="train", cache_dir=cacheDir)

        self.bleuScorer = BLEUScore(n_gram=1)

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = self._getOptions(sample)
        answer = sample["answer"] # Answer is the index of the correct option

        formattedQuestion = f"{question}\n"
        formattedQuestion += "\n".join(options) + "\n"
        formattedQuestion += "What is the correct answer?"

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
        options = [f"{idx}: {choice}." for idx, choice in enumerate(choices)]
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