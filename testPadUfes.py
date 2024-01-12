from multimedbench.utils import Params
from multimedbench.utils import Params
from multimedbench.engine import MMB

# Test the class
if __name__ == "__main__":
    params = Params(True, 42, 64)

    def batcher(prompts):
        return ["Actinic Keratosis" for _ in range(len(prompts))]

    engine = MMB(params=params, batcher=batcher)

    print(engine.eval("Pad-UFES-20")[0])