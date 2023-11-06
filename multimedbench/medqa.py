import logging
from datasets import load_dataset
from utils import Benchmark, batchSampler, Params
from tqdm import tqdm
import time
import math

class MedQA(Benchmark):
    def __init__(self, seed=1111):
        logging.info("***** Benchmarking : MedQA *****")
        self.seed = seed

        # Get the dataset
        self.dataset = load_dataset("bigbio/med_qa", split="test", cache_dir="data/medqa")
        
        # Create the prompt base
        trainDataset = load_dataset("bigbio/med_qa", split="train", cache_dir="data/medqa")
        self.prompt = "\n\n".join(self.format_question(trainDataset[i], prompt=True) for i in range(2)) + "\n\n"

    def run(self, params: Params, batcher):
        correct_answers = 0
        total_answers = 0

        # Run the batcher for all data split in chunks
        for batch in tqdm(batchSampler(self.dataset, params.batch_size), total=math.ceil(len(self.dataset) / params.batch_size), desc="Running inference"):
            batchPrompts = [(self.prompt + self.format_question(sample), {}) for sample in batch]

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                if self.isValid(answer, self.getCorrectAnswer(batch[idx])):
                    correct_answers += 1
                total_answers += 1

        # Compute the scores
        return {"accuracy": correct_answers / total_answers}


    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer_idx"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}' for option in options])
        formattedQuestion += f"\nAnswer:\n{answer if prompt else ''}"
        return formattedQuestion
    
    def getCorrectAnswer(self, sample):
        return sample["answer_idx"]
    
    def isValid(self, pred:str, gold:str):
        pred = pred.lower().strip().replace("\n", "")
        gold = gold.lower().strip().replace("\n", "")
        return pred == gold
    
class PubMedQA(MedQA):
    def __init__(self, seed=1111):
        logging.info("***** Benchmarking : PubMedQA *****")
        self.seed = seed

        # Get the dataset
        self.dataset = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold1_bigbio_qa", split="test", cache_dir="data/pubmedqa")
        
        # Prepare the prompt base
        trainDataset = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold1_bigbio_qa", split="train", cache_dir="data/pubmedqa")
        self.prompt = "\n\n".join(self.format_question(trainDataset[i], prompt=True) for i in range(2)) + "\n\n"

    def getCorrectAnswer(self, sample):
        return sample["answer"][0]

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        answer = sample["answer"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options: yes, no or maybe"
        formattedQuestion += f"\nAnswer:\n{answer[0] if prompt else ''}"
        return formattedQuestion
    

class MedMCQA(MedQA):
    def __init__(self, seed=1111):
        logging.info("***** Benchmarking : MedMCQA *****")
        self.seed = seed

        # Get the dataset
        self.dataset = load_dataset("medmcqa", name="pubmed_qa_labeled_fold1_bigbio_qa", split="validation", cache_dir="data/medmcqa")

        # Prepare the prompt base
        trainDataset = load_dataset("medmcqa", name="pubmed_qa_labeled_fold1_bigbio_qa", split="train", cache_dir="data/medmcqa")
        self.prompt = "\n\n".join(self.format_question(trainDataset[i], prompt=True) for i in range(2)) + "\n\n"


    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = [f"A: {sample['opa']}", f"B: {sample['opb']}", f"C: {sample['opc']}", f"D: {sample['opd']}"]
        answer = sample["cop"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options:\n" + "\n".join(options)
        formattedQuestion += f"\nAnswer:\n{chr(ord('a') + answer - 1).upper() if prompt else ''}"
        return formattedQuestion
    
    def getCorrectAnswer(self, sample):
        return chr(ord('a') + sample["cop"] - 1).upper()




# Test the benchmark
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)


    benchmark = MedQA()

    def batcher(batch):
        time.sleep(0.01)
        return ["A" for _ in batch]

    results = benchmark.run(Params(True, 42, 16), batcher)
    print(f"Test results MedQA: {results}")


    benchmark = PubMedQA()

    def batcher(batch):
        time.sleep(0.01)
        return ["yes" for _ in batch]

    results = benchmark.run(Params(True, 42, 16), batcher)
    print(f"Test results PubMedQA: {results}")


    benchmark = MedMCQA()
    
    def batcher(batch):
        time.sleep(0.01)
        return ["A" for _ in batch]

    results = benchmark.run(Params(True, 42, 16), batcher)
    print(f"Test results MedMCQA: {results}")

