from multimedbench.utils import Params
from multimedbench.engine import MMB

from multimedbench.utils import CompositeMetric
import pickle

import dill

# answers = ["test text"]

# df = pd.DataFrame(columns=["Report Impression"], data=answers)
# print(df)
# labels = label("chexbert.pth", df)
# print(labels)

# raise Exception


params = Params(True, 42, 64)


def batcher(prompts):
    return ["yes it is a good report" for _ in range(len(prompts))]


engine = MMB(params=params, batcher=batcher)

print(engine.eval("MIMIC-CXR")[0])

