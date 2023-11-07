from dataclasses import dataclass


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Benchmark():
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

