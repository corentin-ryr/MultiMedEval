
from multimedbench.utils import Params, fileWriterFactory, Benchmark

from multimedbench.qa import MedQA, PubMedQA, MedMCQA
from multimedbench.vqa import VQA_RAD, Path_VQA
from multimedbench.mimic import MIMIC_CXR_reportgen
import json
import os
import gdown
import sys



TASKS:dict[str, Benchmark] = {
    "MedQA": MedQA,
    "PubMedQA": PubMedQA,
    "MedMCQA": MedMCQA,
    "MIMIC-CXR": MIMIC_CXR_reportgen,
    "VQA-RAD": VQA_RAD,
    "Path-VQA": Path_VQA
}


class MMB(object):
    def __init__(self, params:Params, batcher, prepare=None):
        self.params = params
        print(f"\n\nRunning MultiMedBenchmark with {self.params}")

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        if not os.path.exists(params.run_name):
            os.mkdir(params.run_name)

        self._prepare_radgraph()
        self._prepare_chexbert()



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
                    fileWriterFactory(result["type"])(result["value"], f"{self.params.run_name}/{result['name']}")


            return self.results

        assert name in TASKS, str(name) + ' not in ' + str(self.list_tasks)

        self.evaluation:Benchmark = TASKS[name](seed=self.params.seed, engine=self)
        taskResult = self.evaluation.run(self.params, self.batcher)

        return taskResult
    
        
    def _prepare_radgraph(self):
        # Open the MedMD_config json file and get the download location for radgraph
        with open("MedMD_config.json", "r") as f:
            output = json.load(f)["RadGraph"]["dlLocation"]

        if not os.path.exists(os.path.join(output, "scorers")):
            gdown.download("https://drive.google.com/uc?id=1koePS_rgP5_zNUeqnQgdQ89nQEolTEbR", output, quiet=False)

            # Unzip the archive and delete the archive
            import zipfile
            with zipfile.ZipFile(os.path.join(output, "scorers.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(output, "scorers"))
            os.remove(os.path.join(output, "scorers.zip"))
        else:
            print("RadGraph already downloaded")

        # Add the RadGraph to the path
        sys.path.append(output)

        try:
            from scorers.RadGraph.RadGraph import RadGraph # It is normal that the import is not found by the IDE because it will be downloaded and installed at runtime
        except Exception as e:
            print("There was an error during the download and install of RadGraph")
            raise e
        
        self.radgraph = RadGraph(reward_level="partial")

    def _prepare_chexbert(self):
        # Download the Chexbert checkpoint from https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
        with open("MedMD_config.json", "r") as f:
            output = json.load(f)["CheXBert"]["dlLocation"]

        if not os.path.exists(os.path.join(output, "chexbert.pth")):
            os.makedirs(output, exist_ok=True)
            gdown.download("https://stanfordmedicine.app.box.com/shared/static/c3stck6w6dol3h36grdc97xoydzxd7w9", os.path.join(output, "chexbert.pth"), quiet=False)
        else:
            print("Chexbert already downloaded")


