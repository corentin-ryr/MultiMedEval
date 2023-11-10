from dataclasses import dataclass
import string

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Benchmark():
    def __init__(self) -> None:
        pass

    def do_prepare(self, params, prepare):
        pass

    def run(self, params, batcher):
        pass


@dataclass
class Params:
    usepytorch: bool = True
    seed: int = 1111
    batch_size: str = 128


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