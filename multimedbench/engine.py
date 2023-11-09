
from multimedbench import utils

from multimedbench.medqa import MedQA, PubMedQA, MedMCQA
from multimedbench.mimic import MIMIC_CXR_classification
import json


TASKS:dict[str, utils.Benchmark] = {
     "MedQA": MedQA,
     "PubMedQA": PubMedQA,
     "MedMCQA": MedMCQA,
     "MIMIC-CXR": MIMIC_CXR_classification
}


class MMB(object):
    def __init__(self, params:utils.Params, batcher, prepare=None):
        self.params = params
        print(f"\n\nRunning MultiMedBenchmark with {self.params}")

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None



    def eval(self, name:str|list[str]):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {}
            for x in name:
                currentResults = self.eval(x)
                self.results[x] = currentResults
                print(currentResults)

            # Write the results to a file
            with open("results.json", "w") as f: json.dump(self.results, f)

            return self.results

        assert name in TASKS, str(name) + ' not in ' + str(self.list_tasks)

        self.evaluation:utils.Benchmark = TASKS[name](seed=self.params.seed)
        self.evaluation.do_prepare(self.params, self.prepare)
        taskResult = self.evaluation.run(self.params, self.batcher)

        return taskResult