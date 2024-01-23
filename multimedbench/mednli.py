from multimedbench.utils import Benchmark, Params
from tqdm import tqdm
import os
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

class MedNLI(Benchmark):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedNLI"
        self.modality = "General medicine"
        self.task = "NLI"
     
        self.path = self.engine.getConfig()["physionetCacheDir"]["path"]
        self._generate_dataset()

        testSet = pd.read_json(path_or_buf=os.path.join(self.path, "mli_test_v1.jsonl"), lines=True)

        self.options = testSet["gold_label"].unique().tolist()

        testSet = testSet[["sentence1", "sentence2", "gold_label"]]
        self.dataset = Dataset.from_pandas(testSet)

        if self.fewshot:
            trainSet = pd.read_json(path_or_buf=os.path.join(self.path, "mli_train_v1.jsonl"), lines=True)
            trainSet = trainSet[["sentence1", "sentence2", "gold_label"]]
            self.trainDataset = Dataset.from_pandas(trainSet)


    def run(self, params: Params, batcher):
        print(f"***** Benchmarking : {self.taskName} *****")

        correct_answers = 0
        total_answers = 0

        answersLog = []

        dataloader = DataLoader(self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x)
        for batch in tqdm(
            dataloader,
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                if self.fewshot:
                    batchPrompts.append((self.getPrompt()[0] + text, self.getPrompt()[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                gold = self.getCorrectAnswer(batch[idx])
                pred = self.getPredictedAnswer(answer, batch[idx])
                if pred == gold:
                    correct_answers += 1
                total_answers += 1

                answersLog.append((self.getCorrectAnswer(batch[idx], fullText=True), answer, pred == gold))
            
        metrics = {"accuracy": correct_answers / total_answers}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]
    

    def format_question(self, sample, prompt=False):
        formattedQuestion = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}\n"
        formattedQuestion += "Determine the logical relationship between these two sentences. Does the second sentence logically follows from the first (entailment), contradicts the first (contradiction), or if there is no clear logical relationship between them (neutral)?"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            formattedAnswer = "The logical relationship is " + sample["gold_label"] + "."
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        return sample["gold_label"]

    def getPredictedAnswer(self, pred: str, sample):
        # Check if the prediction contains each of the options
        scores = []
        for option in self.options:
            scores.append(option in pred)
        
        if sum(scores) != 1:
            return "Invalid answer"
        
        return self.options[scores.index(True)]



    def _generate_dataset(self):
        # Check if the path already exists and if so return
        # Path /shares/menze.dqbm.uzh/corentin/physionetCacheDir/physionet.org/files/mednli/1.0.0
        if os.path.exists(
            os.path.join(self.path, "physionet.org", "files", "mednli", "1.0.0", "mli_test_v1.jsonl")
        ):
            self.path = os.path.join(self.path, "physionet.org", "files", "mednli", "1.0.0")
            return

        os.makedirs(self.path, exist_ok=True)

        url = "https://physionet.org/files/vindr-mammo/1.0.0/"
        username, password = self.engine.getPhysioNetCredentials()
        response = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)

        if response.status_code == 200:
            with open(self.path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Download successful. File saved to: {self.path}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            print(response.text)

            raise Exception("Failed to download the dataset")

        self.path = os.path.join(self.path, "physionet.org", "files", "mednli", "1.0.0")

        