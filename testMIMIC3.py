from multimedbench.utils import Params
from multimedbench.engine import MMB

params = Params( 42, 64)


def batcher(prompts):
    return ["" for _ in range(len(prompts))]


engine = MMB(params=params, batcher=batcher)

print(engine.eval("MIMIC-III")[0])