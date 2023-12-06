
from multimedbench import utils

from multimedbench.qa import MedQA, PubMedQA, MedMCQA
from multimedbench.vqa import VQA_RAD, Path_VQA
from multimedbench.mimic import MIMIC_CXR_classification
import json
import os
import gdown


TASKS:dict[str, utils.Benchmark] = {
    "MedQA": MedQA,
    "PubMedQA": PubMedQA,
    "MedMCQA": MedMCQA,
    "MIMIC-CXR": MIMIC_CXR_classification,
    "VQA-RAD": VQA_RAD,
    "Path-VQA": Path_VQA
}


class MMB(object):
    def __init__(self, params:utils.Params, batcher, prepare=None):
        self.params = params
        print(f"\n\nRunning MultiMedBenchmark with {self.params}")

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        if not os.path.exists(params.run_name):
            os.mkdir(params.run_name)

        self._prepare_radgraph()



    def eval(self, name:str|list[str]):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {}
            for x in name:
                currentResults = self.eval(x)
                self.results[x] = currentResults
                print(f"Done task {x}")

                # Write to files
                for result in currentResults:
                    if result["type"] == "json": print(result["value"])
                    utils.fileWriterFactory(result["type"])(result["value"], f"{self.params.run_name}/{result['name']}")


            return self.results

        assert name in TASKS, str(name) + ' not in ' + str(self.list_tasks)

        self.evaluation:utils.Benchmark = TASKS[name](seed=self.params.seed)
        taskResult = self.evaluation.run(self.params, self.batcher)

        return taskResult
    
        
    def _prepare_radgraph(self):
        # Open the MedMD_config json file and get the download location for radgraph
        with open("multimedbench/scorers/RadGraph/MedMD_config.json", "r") as f:
            output = json.load(f)["radgraph"]["dlLocation"]

        gdown.download("https://drive.google.com/uc?id=1koePS_rgP5_zNUeqnQgdQ89nQEolTEbR", output, quiet=False)

        # Unzip the archive and delete the archive
        utils.unzip(output, output)
        os.remove(output)



        from scorers.RadGraph.RadGraph import RadGraph
        self.radgraph = RadGraph(reward_level="partial")