from datasets import load_dataset
from multimedbench.qa import QA, STOPWORDS
from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore

class VQA_RAD(QA):
    def __init__(self, data_folder="data", seed=1111) -> None:
        super().__init__(data_folder, seed)
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", cache_dir=f"{data_folder}/vqa-rad")
        self.trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train", cache_dir=f"{data_folder}/vqa-rad")

        self.rouge = ROUGEScore()

    def format_question(self, sample, prompt=False):
        print(sample)


        formattedQuestion = f"On this picture <img0>, {sample['question']}"
        formattedAnswer = f"The answer is {sample['answer']}"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, {"img0": sample["image"]})

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0: return False
        
        answer = self.cleanStr(sample["answer"])
        return self.rouge(pred, answer)["rouge1_fmeasure"].item() > 0.5
    
class Path_VQA(QA):
    def __init__(self, data_folder="data", seed=1111) -> None:
        super().__init__(data_folder, seed)
        self.dataset = load_dataset("flaviagiammarino/path-vqa", split="test", cache_dir=f"{data_folder}/path-vqa")
        self.trainDataset = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir=f"{data_folder}/path-vqa")

        self.rouge = ROUGEScore()

    def format_question(self, sample, prompt=False):
        print(sample)


        formattedQuestion = f"On this picture <img0>, {sample['question']}"
        formattedAnswer = f"The answer is {sample['answer']}"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, {"img0": sample["image"]})

    def getCorrectAnswer(self, sample):
        return sample["answer"].lower().strip()

    def isValid(self, pred: str, sample):
        pred = self.cleanStr(pred)
        if len(pred) == 0: return False
        
        answer = self.cleanStr(sample["answer"])
        return self.rouge(pred, answer)["rouge1_fmeasure"].item() > 0.5

        
    

    