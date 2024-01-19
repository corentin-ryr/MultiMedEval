from multimedbench.utils import Benchmark, batchSampler, Params
from datasets import load_dataset
from tqdm import tqdm
import math
import random
from multimedbench.utils import remove_punctuation
import os
import requests
from requests.auth import HTTPBasicAuth

class MedNLI(Benchmark):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.taskName = "MedNLI"
        self.modality = "General medicine"
        self.task = "NLI"
     
        self.path = self.engine.getConfig()["physionetCacheDir"]["path"]
        self._generate_dataset()

        self.options = ['entailment', 'neutral', 'contradiction']



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
                gold = self.getCorrectAnswer(batch[idx])
                pred = self.getPredictedAnswer(answer, batch[idx])
                if pred == gold:
                    correct_answers += 1
                total_answers += 1

                answersLog.append((self.getCorrectAnswer(batch[idx], fullText=True), answer, pred, gold, pred == gold))
            
            break

        # TODO: add others metrics such as AUC, F1...
        metrics = {"accuracy": correct_answers / total_answers}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]


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

    def __len__(self):
        return len(self.dataset)
    

    def format_question(self, sample, prompt=False):
        print(sample)
        raise Exception("Not implemented")

        

        formattedQuestion = f"{question}\n"
        formattedQuestion += (
            "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}.' for option in self.options]) + "\n"
        )
        formattedQuestion += "What is the correct answer?"

        question = [{"role": "user", "content": formattedQuestion}]
        if prompt:
            formattedAnswer = "The answer is " + sample["answer_idx"] + "."
            question.append({"role": "assistant", "content": formattedAnswer})

        return (question, [])

    def getCorrectAnswer(self, sample, fullText=False):
        pass

    def getPredictedAnswer(self, pred: str, sample):
        pass

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(
            os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0", "finding_annotations.csv")
        ):
            self.path = os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0")
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

        self.path = os.path.join(self.path, "physionet.org", "files", "vindr-mammo", "1.0.0")

        