from multimedbench.utils import Benchmark, Params, batchSampler
import os
import urllib.request
import zipfile
from tqdm import tqdm
import math
import pandas as pd
import shutil



class Pad_UFES_20(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dataFolder = "/home/croyer/data/pad_ufes_20"
        # Check if the folder countains the zip file
        if not os.path.exists(f"{dataFolder}.zip"):
            # Download the file
            print("Downloading the dataset...")
            url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, f"{dataFolder}.zip", reporthook=lambda x, y, z: t.update(y))

            # Extract the file
            print("Extracting the dataset...")
            with zipfile.ZipFile(f"{dataFolder}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{dataFolder}")

            print("Extracting the images...")
            for file in os.listdir(f"{dataFolder}/images"):
                if not file.endswith(".zip"): continue
                with zipfile.ZipFile(f"{dataFolder}/images/{file}", 'r') as zip_ref:
                    zip_ref.extractall(f"{dataFolder}/images")
                    os.remove(f"{dataFolder}/images/{file}")
            
            print("Copying the images...")
            for file in os.listdir(f"{dataFolder}/images"):
                if not os.path.isdir(f"{dataFolder}/images/{file}"): continue
                for image in os.listdir(f"{dataFolder}/images/{file}"):
                    shutil.copyfile(f"{dataFolder}/images/{file}/{image}", f"{dataFolder}/images/{image}")
                    os.remove(f"{dataFolder}/images/{file}/{image}")
                
                os.rmdir(f"{dataFolder}/images/{file}")


        self.dataset = pd.read_csv(f"{dataFolder}/metadata.csv")
        print(self.dataset.columns)

        images = os.listdir(f"{dataFolder}/images")
        print(len(images))


    def run(self, params:Params, batcher):
        
        correct_answers = 0
        total_answers = 0

        # Run the batcher for all data split in chunks
        for batch in tqdm(batchSampler(self.dataset, params.batch_size), total=math.ceil(len(self.dataset) / params.batch_size), desc="Running inference"):
            print(batch)
            batchPrompts = [self.format_question(batch.iloc[i]) for i in range(len(batch))]

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
        question = f"Given <img> and the following condition of the patient:"

        options = "Options:\n" + "\n".join(["Basal Cell Carcinoma (BCC)", "Squamous Cell Carcinoma (SCC)", "Actinic Keratosis (ACK)", 
                           "Seborrheic Keratosis (SEK)", "Bowen’s disease (BOD)", "Melanoma (MEL)", "Nevus (NEV)"])

        image = None

        return (sample["cancer_history"], {})
    
    def getCorrectAnswer(self, sample):
        return sample["diagnostic"]
    

# Test the class
if __name__ == "__main__":
    params = Params(True, 42, 64)

    def batcher(prompts):
        return ["Basal Cell Carcinoma" for _ in len(prompts)]

    pad_ufes_20 = Pad_UFES_20()
    print(pad_ufes_20.run(params, None))