from multimedbench.utils import Params
from multimedbench.engine import MMB


params = Params(True, 42, 64)

def batcher(prompts):
    return ["answer b is correct" for _ in range(len(prompts))]

engine = MMB(params=params, batcher=batcher)

print(engine.eval("MedMCQA")[0])