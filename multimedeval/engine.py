from multimedeval.utils import EvalParams, fileWriterFactory, Benchmark, SetupParams

from multimedeval.qa import MedQA, PubMedQA, MedMCQA
from multimedeval.vqa import VQA_RAD, Path_VQA, SLAKE
from multimedeval.mimic import MIMIC_CXR_reportgen
from multimedeval.imageClassification import (
    MIMIC_CXR_ImageClassification,
    VinDr_Mammo,
    Pad_UFES_20,
    CBIS_DDSM_Mass,
    CBIS_DDSM_Calcification,
)
from multimedeval.mimic_iii import MIMIC_III
from multimedeval.mednli import MedNLI
from multimedeval.mnist import (
    MNIST_Oct,
    MNIST_Path,
    MNIST_Blood,
    MNIST_Breast,
    MNIST_Derma,
    MNIST_OrganC,
    MNIST_OrganS,
    MNIST_Pneumonia,
    MNIST_Retina,
    MNIST_Tissue,
)
import os
import gdown
from multimedeval.tqdm_loggable import tqdm_logging
import getpass
import nltk
from multimedeval.visualization import BenchmarkVisualizer
from collections.abc import Callable
from radgraph import F1RadGraph
from multimedeval.chexbert.label import encode, encode, label
from dataclasses import asdict
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASKS: set[Benchmark] = {
    MedQA,
    PubMedQA,
    MedMCQA,
    VQA_RAD,
    Path_VQA,
    SLAKE,
    MIMIC_CXR_reportgen,  # Setup not tested
    MIMIC_III,
    MedNLI,
    MIMIC_CXR_ImageClassification,  # Setup not tested
    VinDr_Mammo,  # Setup not tested
    Pad_UFES_20,
    CBIS_DDSM_Mass,
    CBIS_DDSM_Calcification,
    MNIST_Oct,
    MNIST_Path,
    MNIST_Blood,
    MNIST_Breast,
    MNIST_Derma,
    MNIST_OrganC,
    MNIST_OrganS,
    MNIST_Pneumonia,
    MNIST_Retina,
    MNIST_Tissue,
    # # # "MNIST-OrganA": MNIST_OrganA,
    # # # "MNIST-Chest": MNIST_Chest,
}


TASKS_REQUIREMENTS: dict[str, list[str]] = {
    MIMIC_CXR_reportgen: ["RadGraph", "Chexbert"],
    MIMIC_CXR_ImageClassification: ["Chexbert"],
    MIMIC_III: ["RadGraph", "Chexbert"],
}


class MultiMedEval(object):
    def __init__(self):
        self._config: SetupParams = None
        self._physionet_username = None
        self._physionet_password = None

        self.logger = logging.getLogger("MultiMedEval")

        self.tasksReady = {}

        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.nameToTask: dict[str, Benchmark] = {}
        self.nameToRequirements: dict[str, list[str]] = {}
        for taskClass in TASKS:
            benchmark: Benchmark = taskClass(engine=self, logger=self.logger)
            self.nameToTask[benchmark.taskName] = benchmark
            self.nameToRequirements[benchmark.taskName] = (
                TASKS_REQUIREMENTS[taskClass] if taskClass in TASKS_REQUIREMENTS else []
            )

            self.tasksReady[benchmark.taskName] = {"ready": False, "error": "Not setup yet"}

    def setup(self, setupParams: SetupParams, verbose: bool = True):
        self.logger.info(f"Starting the setup of MultiMedEval.")
        self._config = setupParams
        tasksToSkip = []
        # if len(self.getConfig()["tasks_to_prepare"]) > 0:
        #     tasksToSkip = [x for x in self.nameToTask if x not in self.getConfig()["tasks_to_prepare"]]

        progressBar = tqdm_logging(logger=self.logger, total=len(self.nameToTask) + 2, dynamic_ncols=True)
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

        for taskName in self.nameToTask:
            progressBar.set_description(f"Setup {taskName}")
            try:
                if taskName in tasksToSkip:
                    raise Exception(f"Task {taskName} is skipped")
                self.nameToTask[taskName].setup()
            except Exception as e:
                self.tasksReady[taskName] = {"ready": False, "error": str(e)}
            else:
                self.tasksReady[taskName] = {"ready": True}

            progressBar.update(1)
        progressBar.close()

        finalMessage = "End of setup."

        if verbose:
            finalMessage += "\n"
            # Log a table of the tasks and their status
            finalMessage += "Task".ljust(35) + "Status".ljust(20) + "Error"
            for taskName in self.tasksReady:
                error = "No error." if "error" not in self.tasksReady[taskName] else self.tasksReady[taskName]["error"]
                ready = "Ready" if self.tasksReady[taskName]["ready"] else "Problem"
                finalMessage += "\n" + taskName.ljust(35) + ready.ljust(20) + error

        self.logger.info(finalMessage)

        return self.tasksReady

    def eval(self, name: str | list[str], batcher: Callable, evalParams: EvalParams = None):
        if batcher is None:
            raise Exception("The engine was not initialized with a batcher, please provide a batcher to the engine")

        self.evalParams = evalParams if evalParams is not None else EvalParams()
        self.batcher = batcher

        if not os.path.exists(evalParams.run_name):
            os.mkdir(evalParams.run_name)

        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(name, list):
            if len(name) == 0:
                name = list(self.nameToTask.keys())
            self.results = {}
            for x in name:
                currentResults = self.eval(x, batcher, evalParams)
                if currentResults is None:
                    continue
                self.results[x] = currentResults

            return self.results

        if name not in self.nameToTask:
            self.logger.warn(
                f"Task {name} not in {list(self.nameToTask.keys())}",
            )
            return None

        # Check if the requirements are satisfied
        listRequirements = [name] + self.nameToRequirements[name]
        for req in listRequirements:
            if not self.tasksReady[req]["ready"]:
                error = self.tasksReady[req]["error"] if "error" in self.tasksReady[req] else "No error message"

                self.logger.warn(f"Task {name} requires {req} to be ready: {error}")
                return None

        evaluation: Benchmark = self.nameToTask[name]
        taskResult = evaluation.run(self.evalParams, self.batcher)

        # Write to files
        for result in taskResult:
            if result["type"] == "json":
                self.logger.info(result["value"])
                self._writeTotensorboard(result)
            fileWriterFactory(result["type"])(result["value"], f"{self.evalParams.run_name}/{result['name']}")

        self.logger.info(f"Done task {name}")

        return taskResult

    def visualization(self):
        benchmarks = [self.nameToTask[x] for x in self.tasksReady if (self.tasksReady[x]["ready"] and x in self.nameToTask)]
        visualizer = BenchmarkVisualizer(benchmarks)
        visualizer.sunburstModalities()
        visualizer.sunburstTasks()
        visualizer.tableImageClassification()
        visualizer.sankeyDiagram()
        # visualizer.sankeyD3Blocks()

    def getPhysioNetCredentials(self):
        if self._physionet_password is None or self._physionet_username is None:
            self._physionet_username = self.getConfig()["physionet_username"]
            self._physionet_password = self.getConfig()["physionet_password"]
            if not self._physionet_username or not self._physionet_password:
                self.logger.info(
                    "To setup the tasks that use a PhysioNet dataset, the scripts requires the PhysioNet username and password."
                )
                self._physionet_username = input("Enter your username: ")
                self._physionet_password = getpass.getpass("Enter your password: ")

        return self._physionet_username, self._physionet_password

    def getConfig(self) -> dict:
        if self._config is None:
            raise Exception("The engine was not setup, please run the setup method first.")

        return asdict(self._config)

    def _writeTotensorboard(self, results):
        writer = self.evalParams.tensorboardWriter
        if writer is None:
            return

        runName = self.evalParams.run_name
        taskName = results["name"]

        metrics = results["value"]
        for metric in metrics:
            metricValue = metrics[metric]
            writer.add_scalar(
                f"{runName}/{taskName}/{metric}", metricValue, global_step=self.evalParams.tensorboardStep
            )

    def _prepare_radgraph(self):
        device = -1 if self.getConfig()["device"] != "cuda" else 0
        self.radgraph = F1RadGraph(reward_level="partial", cuda=device)

    def _prepare_chexbert(self):
        # Download the Chexbert checkpoint from https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
        output = os.path.join(self.getConfig()["CheXBert_dir"], "chexbert.pth")

        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
            gdown.download(
                "https://stanfordmedicine.app.box.com/shared/static/c3stck6w6dol3h36grdc97xoydzxd7w9",
                output,
                quiet=False,
            )

        self.encoder = encode(output, verbose=False)
        self.labeler = label(output, verbose=False)

    def __len__(self):
        total_len = 0
        for task in self.nameToTask:
            if self.tasksReady[task]["ready"]:
                total_len += len(self.nameToTask[task])

        return total_len