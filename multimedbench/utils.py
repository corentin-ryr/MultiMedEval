from dataclasses import dataclass
import string
from datetime import datetime
import csv
import json
import random


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Benchmark:
    def __init__(self, engine, seed=1111) -> None:
        self.seed = seed
        random.seed(self.seed)
        self.taskName = "None"
        self.engine = engine

    def run(self, params, batcher):
        pass


@dataclass
class Params:
    usepytorch: bool = True
    seed: int = 1111
    batch_size: str = 128
    run_name: str = f"run {datetime.now()}"


def batchSampler(samples, n):
    for i in range(0, len(samples), n):
        # if is a panda dataframe
        if hasattr(samples, "iloc"):
            yield samples.iloc[i : min(i + n, len(samples))]
        else:
            yield [samples[j] for j in range(i, min(i + n, len(samples)))]


def remove_punctuation(input_string: str):
    # Make a translator object to replace punctuation with none
    translator = str.maketrans("", "", string.punctuation)
    # Use the translator
    return input_string.translate(translator)


class RateEstimation:
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
        with open(f"{path}.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)


SUPPORTED_FILETYPES = {"csv": csvWriter, "json": jsonWriter}


def fileWriterFactory(fileType):
    assert fileType in SUPPORTED_FILETYPES, f"{fileType} not supported."

    return SUPPORTED_FILETYPES[fileType]


def exact_entity_token_if_rel_exists_reward(
    hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    # print(hypothesis_relation_token_list)
    # print(reference_relation_token_list)

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score
