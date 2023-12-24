from multimedbench.imageClassification import MIMIC_CXR_ImageClassification
import json
from multimedbench.utils import Params
from multimedbench.engine import MMB

params = Params(True, 42, 64)


def batcher(prompts):
    return ["malignant" for _ in range(len(prompts))]


engine = MMB(params=params, batcher=batcher)

engine.eval(["MIMIC-CXR-ImageClassification"]) #"CBIS-DDSM", "Pad-UFES-20", "VinDr-Mammo", 