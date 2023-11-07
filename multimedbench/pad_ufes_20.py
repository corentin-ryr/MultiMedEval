from utils import Benchmark, Params, batchSampler
import os
import urllib.request
import zipfile
from tqdm import tqdm
import math
import csv
import pandas as pd


class Pad_UFES_20(Benchmark):
    def __init__(self) -> None:
        super().__init__()

        # Check if the folder countains the zip file
        if not os.path.exists("data/pad_ufes_20"):
            # Download the file
            print("Downloading the dataset...")
            url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, "data/pad_ufes_20/pad_ufes_20.zip", reporthook=lambda x, y, z: t.update(y))

            # Extract the file
            print("Extracting the dataset...")
            with zipfile.ZipFile("data/pad_ufes_20/pad_ufes_20.zip", 'r') as zip_ref:
                zip_ref.extractall("data/pad_ufes_20")

        self.dataset = pd.read_csv("data/pad_ufes_20/metadata.csv")
        print(self.dataset.columns)


    def run(self, params:Params, batcher):
        
        correct_answers = 0
        total_answers = 0

        # Run the batcher for all data split in chunks
        for batch in tqdm(batchSampler(self.dataset, params.batch_size), total=math.ceil(len(self.dataset) / params.batch_size), desc="Running inference"):
            batchPrompts = [self.format_question(sample) for sample in batch]

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                if self.isValid(answer, self.getCorrectAnswer(batch[idx])):
                    correct_answers += 1
                total_answers += 1

        # Compute the scores
        return {"accuracy": correct_answers / total_answers}
    
    def format_question(self, sample):
        # Features: smoke,drink,background_father,background_mother,age,pesticide,gender,skin_cancer_history,cancer_history,
        # has_piped_water,has_sewage_system,fitspatrick,region,diameter_1,diameter_2,itch,grew,hurt,changed,bleed,elevation

        return (sample["question"], {})
    
    def getCorrectAnswer(self, sample):
        return sample["diagnostic"]
    

# Test the class
if __name__ == "__main__":
    params = Params(True, 42, 64)

    pad_ufes_20 = Pad_UFES_20()
    print(pad_ufes_20.run(params, None))