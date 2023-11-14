from datasets import load_dataset
from multimedbench.qa import QA


class VQA_RAD(QA):
    def __init__(self) -> None:
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
        trainDataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

    