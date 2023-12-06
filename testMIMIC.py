from multimedbench.mimic import MIMIC_CXR_reportgen
from multimedbench.utils import Params

params = Params(True, 42, 64)

def batcher(prompts):
    return ["yes" for _ in range(len(prompts))]

mimic = MIMIC_CXR_reportgen()

print(mimic.run(params, batcher)[0])