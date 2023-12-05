from dataclasses import dataclass
import string
from datetime import datetime
import csv
import json
import random

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Benchmark():
    def __init__(self, seed=1111) -> None:

        self.seed = seed
        random.seed(self.seed)
        self.taskName = "None"

    def run(self, params, batcher):
        pass


@dataclass
class Params:
    usepytorch: bool = True
    seed: int = 1111
    batch_size: str = 128
    run_name:str = f"run {datetime.now()}"


def batchSampler(samples, n):
    for i in range(0, len(samples), n):
        # if is a panda dataframe
        if hasattr(samples, 'iloc'):
            yield samples.iloc[i:min(i+n, len(samples))]
        else:
            yield [samples[j] for j in range(i, min(i+n, len(samples)))]  

def remove_punctuation(input_string:str):
    # Make a translator object to replace punctuation with none
    translator = str.maketrans('', '', string.punctuation)
    # Use the translator
    return input_string.translate(translator)


class RateEstimation():
    def __init__(self) -> None:
        self.exectimes = []

    def update(self, value):
        self.exectimes.append(value)

    def getRate(self):
        pass



def csvWriter(data, path):
    try:
        with open(f"{path}.csv", "w", newline="") as f:
            spamWriter = csv.writer(f)
            spamWriter.writerows(data)
    except Exception as e:
        print(e)

def jsonWriter(data, path):
    try:
        with open(f"{path}.json", "w") as f: json.dump(data, f)
    except Exception as e:
        print(e)


SUPPORTED_FILETYPES = {"csv": csvWriter, "json": jsonWriter}
def fileWriterFactory(fileType):
    assert fileType in SUPPORTED_FILETYPES, f"{fileType} not supported."

    return SUPPORTED_FILETYPES[fileType]