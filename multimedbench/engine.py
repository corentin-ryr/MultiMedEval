from multimedbench.utils import Params, fileWriterFactory, Benchmark

from multimedbench.qa import MedQA, PubMedQA, MedMCQA
from multimedbench.vqa import VQA_RAD, Path_VQA, SLAKE
from multimedbench.mimic import MIMIC_CXR_reportgen
from multimedbench.imageClassification import MIMIC_CXR_ImageClassification, VinDr_Mammo, Pad_UFES_20, CBIS_DDSM_Mass
from multimedbench.mimic_iii import MIMIC_III
import json
import os
import gdown
import sys
from tqdm import tqdm
import getpass
import nltk
from multimedbench.visualization import BenchmarkVisualizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


TASKS: dict[str, Benchmark] = {
    "MedQA": MedQA,
    "PubMedQA": PubMedQA,
    "MedMCQA": MedMCQA,
    "MIMIC-CXR-ReportGeneration": MIMIC_CXR_reportgen,
    "VQA-RAD": VQA_RAD,
    "Path-VQA": Path_VQA,
    "SLAKE": SLAKE,
    "MIMIC-CXR-ImageClassification": MIMIC_CXR_ImageClassification,
    "VinDr-Mammo": VinDr_Mammo,
    "Pad-UFES-20": Pad_UFES_20,
    "CBIS-DDSM": CBIS_DDSM_Mass,
    "MIMIC-III": MIMIC_III,
}

TASKS_REQUIREMENTS: dict[str, list[str]] = {
    "MedQA": ["MedQA"],
    "PubMedQA": ["PubMedQA"],
    "MedMCQA": ["MedMCQA"],
    "MIMIC-CXR-ReportGeneration": ["MIMIC-CXR-ReportGeneration", "RadGraph", "Chexbert"],
    "VQA-RAD": ["VQA-RAD"],
    "Path-VQA": ["Path-VQA"],
    "SLAKE": ["SLAKE"],
    "MIMIC-CXR-ImageClassification": ["MIMIC-CXR-ImageClassification"],
    "VinDr-Mammo": ["VinDr-Mammo"],
    "Pad-UFES-20": ["Pad-UFES-20"],
    "CBIS-DDSM": ["CBIS-DDSM"],
    "MIMIC-III": ["MIMIC-III", "RadGraph", "Chexbert"],
}


class MMB(object):
    def __init__(self, params: Params, batcher, generateVisualization: bool = False):
        self.params = params
        print(f"\n\nRunning MultiMedBenchmark with {self.params}")

        self.batcher = batcher

        if not os.path.exists(params.run_name):
            os.mkdir(params.run_name)

        print(f"Running the setup")
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.tasksReady = {}

        self._physionet_username = None
        self._physionet_password = None

        progressBar = tqdm(total=len(TASKS) + 2)

        progressBar.set_description(f"Setup RadGraph")
        try:
            self._prepare_radgraph()
        except Exception as e:
            self.tasksReady["RadGraph"] = {"ready": False, "error": str(e)}
        else:
            self.tasksReady["RadGraph"] = {"ready": True}
        progressBar.update(1)
        
        progressBar.set_description(f"Setup Chexbert")
        try:
            self._prepare_chexbert()
        except Exception as e:
            self.tasksReady["Chexbert"] = {"ready": False, "error": str(e)}
        else:
            self.tasksReady["Chexbert"] = {"ready": True}
        progressBar.update(1)


        for taskName in TASKS:
            progressBar.set_description(f"Setup {taskName}")
            try:
                taskBenchmark = TASKS[taskName](seed=self.params.seed, engine=self, fewshot=self.params.fewshot)
            except Exception as e:
                self.tasksReady[taskName] = {"ready": False, "error": str(e)}
            else:
                self.tasksReady[taskName] = {"ready": True, "task": taskBenchmark}

            progressBar.update(1)


        # Print a table of the tasks and their status
        print("\n\n")
        print("Task".ljust(30) + "Status".ljust(30) + "Error")
        for taskName in self.tasksReady:
            error = "" if "error" not in self.tasksReady[taskName] else self.tasksReady[taskName]["error"]
            ready = "Ready" if self.tasksReady[taskName]["ready"] else "Problem"
            print(taskName.ljust(30) + ready.ljust(30) + error)

        if generateVisualization:
            benchmarks = [self.tasksReady[x]["task"] for x in self.tasksReady if (self.tasksReady[x]["ready"] and "task" in self.tasksReady[x])]
            visualizer = BenchmarkVisualizer(benchmarks)
            visualizer.sunburstModalities()


    def eval(self, name: str | list[str]):
        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(name, list):
            self.results = {}
            for x in name:
                currentResults = self.eval(x)
                self.results[x] = currentResults

            return self.results

        assert name in TASKS, str(name) + " not in " + str(TASKS.keys())
        # Check if the requirements are satisfied
        for req in TASKS_REQUIREMENTS[name]:
            if not self.tasksReady[req]["ready"]:
                if "error" in self.tasksReady[req]:
                    error = self.tasksReady[req]["error"]
                else:
                    error = "No error message"
                raise Exception(f"Task {name} requires {req} to be ready: {error}")

        self.evaluation: Benchmark = self.tasksReady[name]["task"]
        taskResult = self.evaluation.run(self.params, self.batcher)

        # Write to files
        for result in taskResult:
            if result["type"] == "json":
                print(result["value"])
            fileWriterFactory(result["type"])(result["value"], f"{self.params.run_name}/{result['name']}")

        print(f"Done task {name}")

        return taskResult

    def _prepare_radgraph(self):
        # Open the MedMD_config json file and get the download location for radgraph
        with open("MedMD_config.json", "r") as f:
            output = json.load(f)["RadGraph"]["dlLocation"]

        if not os.path.exists(os.path.join(output, "scorers")):
            gdown.download("https://drive.google.com/uc?id=1koePS_rgP5_zNUeqnQgdQ89nQEolTEbR", output, quiet=False)

            # Unzip the archive and delete the archive
            import zipfile

            with zipfile.ZipFile(os.path.join(output, "scorers.zip"), "r") as zip_ref:
                zip_ref.extractall(os.path.join(output, "scorers"))
            os.remove(os.path.join(output, "scorers.zip"))
        # else:
        #     print("RadGraph already downloaded")

        # Add the RadGraph to the path
        sys.path.append(output)

        try:
            from scorers.RadGraph.RadGraph import (
                RadGraph,
            )  # It is normal that the import is not found by the IDE because it will be downloaded and installed at runtime
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
            gdown.download(
                "https://stanfordmedicine.app.box.com/shared/static/c3stck6w6dol3h36grdc97xoydzxd7w9",
                os.path.join(output, "chexbert.pth"),
                quiet=False,
            )
        # else:
        #     print("Chexbert already downloaded")

    def getPhysioNetCredentials(self):
        if self._physionet_password is None or self._physionet_username is None:
            print("To setup tasks requiring a PhysioNet dataset, the scripts requires the PhysioNet username and password.")
            self._physionet_username = input("Enter your username: ")
            self._physionet_password = getpass.getpass("Enter your password: ")

        return self._physionet_username, self._physionet_password