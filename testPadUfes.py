from multimedbench.utils import Params
from multimedbench.pad_ufes_20 import Pad_UFES_20


# Test the class
if __name__ == "__main__":
    params = Params(True, 42, 64)

    def batcher(prompts):
        return ["Basal Cell Carcinoma" for _ in len(prompts)]

    pad_ufes_20 = Pad_UFES_20()
    print(pad_ufes_20.run(params, None))