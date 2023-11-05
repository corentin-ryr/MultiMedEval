import logging
from datasets import load_dataset
from utils import Benchmark, batchSampler, Params


class MedQA(Benchmark):
    def __init__(self, task_path, seed=1111):
        logging.debug("***** Benchmarking : MedQA *****\n\n")
        self.seed = seed

        # Get the dataset
        self.dataset = load_dataset("bigbio/med_qa", split="test", cache_dir="data/medqa")

        trainDataset = load_dataset("bigbio/med_qa", split="train", cache_dir="data/medqa")
        self.prompt = "\n\n".join(self.format_question(trainDataset[i], prompt=True) for i in range(2)) + "\n\n"

    def do_prepare(self, params, prepare):
        # Call the prepare function if necessary (it shouldn't)
        return prepare(params, [])

    def run(self, params: Params, batcher):
        # Run the batcher for all data split in chunks
        for batch in batchSampler(self.dataset, params.batch_size):
            batch = [(self.prompt + self.format_question(sample), {}) for sample in batch]

            answers = batcher(batch)

            for idx, answer in enumerate(answers):
                if 

        # Compute the scores

        return {"devpearson": devpr, "pearson": pr, "spearman": sr, "mse": se, "yhat": yhat, "ndev": len(devA), "ntest": len(testA)}

    def format_question(self, sample, prompt=False):
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer_idx"]

        formattedQuestion = f"Question: {question}\n"
        formattedQuestion += "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}' for option in options])
        formattedQuestion += f"\nAnswer:\n{answer if prompt else ''}"

        return formattedQuestion


# Test the benchmark
if __name__ == "__main__":
    benchmark = MedQA("MedQA")

    def batcher():
        pass

    benchmark.run(Params(True, 42, 2), batcher)

