from multimedbench.imageClassification import MIMIC_CXR_ImageClassification
import json
from multimedbench.utils import Params
from multimedbench.engine import MMB

params = Params(True, 42, 64)


def batcher(prompts):
    return ["yes it is a good report" for _ in range(len(prompts))]


engine = MMB(params=params, batcher=batcher)

print(engine.eval("MIMIC-III")[0])