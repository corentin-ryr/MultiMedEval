
import utils

from medqa import MedQA
from mimic import MIMIC_CXR_classification


TASKS:dict[str, utils.Benchmark] = {
     "MedQA": MedQA,
     "MIMIC-CXR": MIMIC_CXR_classification
}


class MMB(object):
    def __init__(self, params:utils.Params, batcher, prepare=None):
        self.params = params
        print(self.params)

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None



    def eval(self, name:str|list[str]):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in TASKS, str(name) + ' not in ' + str(self.list_tasks)

        self.evaluation:utils.Benchmark = TASKS[name](tpath + '/downstream/' + name, seed=self.params.seed)

        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results


# Test the MMB class
if __name__ == "__main__":
    params = utils.Params(True, 42, 64)

    mmb = MMB(params, None)