from multimedbench.mimic import MIMIC_CXR_reportgen
from multimedbench.utils import Params
from multimedbench.engine import MMB

import pandas as pd
from multimedbench.chexbert.label import label

# answers = ["test text"]

# df = pd.DataFrame(columns=["Report Impression"], data=answers)
# print(df)
# labels = label("chexbert.pth", df)
# print(labels)

# raise Exception

params = Params(True, 42, 64)

def batcher(prompts):
    return ["yes" for _ in range(len(prompts))]

engine = MMB(params=params, batcher=batcher)

print(engine.eval("MIMIC-CXR")[0])

