from multimedbench.medqa import PubMedQA
import time
from multimedbench.utils import Params

if __name__ == "__main__":
    benchmark = PubMedQA()

    def batcher(batch):
        time.sleep(0.01)
        return ["yes" for _ in batch]

    results = benchmark.run(Params(True, 42, 16), batcher)
    print(f"Test results PubMedQA: {results}")