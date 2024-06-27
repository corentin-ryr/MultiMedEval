import getpass
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict

import gdown
import nltk
from radgraph import F1RadGraph
from torch.utils.data import DataLoader

from multimedeval.chexbert.label import encode, label
from multimedeval.dynamicDatasets import findDatasets
from multimedeval.imageClassification import (
    CBIS_DDSM_Calcification,
    CBIS_DDSM_Mass,
    MIMIC_CXR_ImageClassification,
    Pad_UFES_20,
    VinDr_Mammo,
)
from multimedeval.mednli import MedNLI
from multimedeval.mimic import MIMIC_CXR_reportgen
from multimedeval.mimic_iii import MIMIC_III
from multimedeval.mnist import (
    MNIST_Blood,
    MNIST_Breast,
    MNIST_Derma,
    MNIST_Oct,
    MNIST_OrganC,
    MNIST_OrganS,
    MNIST_Path,
    MNIST_Pneumonia,
    MNIST_Retina,
    MNIST_Tissue,
)
from multimedeval.qa import MMLU, MedMCQA, MedQA, PubMedQA
from multimedeval.tqdm_loggable import tqdm_logging
from multimedeval.utils import (
    Benchmark,
    EvalParams,
    EvaluationOutput,
    SetupParams,
    fileWriterFactory,
)
from multimedeval.visualization import BenchmarkVisualizer
from multimedeval.vqa import SLAKE, VQA_RAD, DiffVQA, Path_VQA

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASKS: set[Benchmark] = {
    MedQA,
    PubMedQA,
    MedMCQA,
    VQA_RAD,
    Path_VQA,
    DiffVQA,
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
    MMLU,
    # # # "MNIST-OrganA": MNIST_OrganA,
    # # # "MNIST-Chest": MNIST_Chest,
}


TASKS_REQUIREMENTS: dict[str, list[str]] = {
    MIMIC_CXR_reportgen: ["RadGraph", "Chexbert"],
    MIMIC_CXR_ImageClassification: ["Chexbert"],
    MIMIC_III: ["RadGraph", "Chexbert"],
    DiffVQA: ["MIMIC-CXR Report Generation"],
}


class MultiMedEval(object):
    def __init__(self, logger: logging.Logger = None):
        self._config: SetupParams = None
        self._physionet_username = None
        self._physionet_password = None

        if logger is None:
            self.logger = logging.getLogger("MultiMedEval")
        else:
            self.logger = logger

        dynamicDatasets = findDatasets()
        TASKS.update(dynamicDatasets)

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

            self.tasksReady[benchmark.taskName] = {
                "ready": False,
                "error": "Not setup yet",
            }

    def setup(self, setupParams: SetupParams, verbose: bool = True):
        self.logger.info(f"Starting the setup of MultiMedEval.")
        self._config = setupParams
        tasksToSkip = []
        # if len(self.getConfig()["tasks_to_prepare"]) > 0:
        #     tasksToSkip = [x for x in self.nameToTask if x not in self.getConfig()["tasks_to_prepare"]]

        progressBar = tqdm_logging(
            logger=self.logger, total=len(self.nameToTask) + 2, dynamic_ncols=True
        )
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
                error = (
                    "No error."
                    if "error" not in self.tasksReady[taskName]
                    else self.tasksReady[taskName]["error"]
                )
                ready = "Ready" if self.tasksReady[taskName]["ready"] else "Problem"
                finalMessage += "\n" + taskName.ljust(35) + ready.ljust(20) + error

        self.logger.info(finalMessage)

        return self.tasksReady

    def eval(
        self, name: str | list[str], batcher: Callable, evalParams: EvalParams = None
    ):
        if batcher is None:
            raise Exception(
                "The engine was not initialized with a batcher, please provide a batcher to the engine"
            )

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
                taskMetrics = self.eval(x, batcher, evalParams)
                if taskMetrics is None:
                    continue
                self.results[x] = taskMetrics

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
                error = (
                    self.tasksReady[req]["error"]
                    if "error" in self.tasksReady[req]
                    else "No error message"
                )

                self.logger.warn(f"Task {name} requires {req} to be ready: {error}")
                return None

        evaluation: Benchmark = self.nameToTask[name]

        predictions = self._run_inference(evaluation)
        taskResult: EvaluationOutput = evaluation.evaluate(predictions)

        if taskResult.answer_log is not None:
            fileWriterFactory("csv")(
                taskResult.answer_log,
                f"{self.evalParams.run_name}/{evaluation.taskName}",
            )

        try:
            with open(f"{self.evalParams.run_name}/results.json", "r") as f:
                metrics = json.load(f)
        except IOError:
            metrics = {}

        metrics[evaluation.taskName] = taskResult.metrics
        fileWriterFactory("json")(metrics, f"{self.evalParams.run_name}/results")

        self.logger.info(f"Done task {name}")

        return taskResult.metrics

    def _run_inference(self, task: Benchmark):
        self.logger.info(
            f"======================== Running inference on {task.taskName} ========================"
        )

        dataloader = self.get_dataloader(task)
        kwargs_format_question = (
            {"include_indication": self.evalParams.mimic_cxr_include_indication_section}
            if task.taskName == "MIMIC-CXR Report Generation"
            else {}
        )

        predictions = []
        for batch in tqdm_logging(self.logger, dataloader, desc="Running inference"):
            batchPrompts = []
            for el in batch:
                sample = el["sample"]
                text, img = task.format_question(sample, **kwargs_format_question)
                if self.evalParams.fewshot and task.getPrompt() is not None:
                    batchPrompts.append(
                        (task.getPrompt()[0] + text, task.getPrompt()[1] + img)
                    )
                else:
                    batchPrompts.append((text, img))

            answers = self.batcher(batchPrompts)

            for el, answer in zip(batch, answers):
                predictions.append({"idx": el["idx"], "answer": answer})

        return predictions

    def visualization(self):
        benchmarks = [
            self.nameToTask[x]
            for x in self.tasksReady
            if (self.tasksReady[x]["ready"] and x in self.nameToTask)
        ]
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
            raise Exception(
                "The engine was not setup, please run the setup method first."
            )

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
                f"{runName}/{taskName}/{metric}",
                metricValue,
                global_step=self.evalParams.tensorboardStep,
            )

    def _prepare_radgraph(self):
        # Check if deepspeed is installed and initialized
        try:
            from deepspeed.comm.comm import is_initialized

            # Test if deepspeed is initialized
            if not is_initialized():
                raise Exception("Deepspeed is not initialized.")
        except:
            pass
        else:
            raise Exception("Deepspeed is initialized.")

        device = -1 if self.getConfig()["device"] != "cuda" else 0
        self.radgraph = F1RadGraph(
            reward_level="partial", cuda=device, model_type="radgraph"
        )

    def _prepare_chexbert(self):
        # Download the Chexbert checkpoint from https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
        path = self.getConfig()["CheXBert_dir"]

        if path is None:
            raise Exception("CheXBert_dir is not set in the config file.")

        output = os.path.join(path, "chexbert.pth")

        if not os.path.exists(output):
            os.makedirs(path, exist_ok=True)
            gdown.download(
                "https://stanfordmedicine.app.box.com/shared/static/c3stck6w6dol3h36grdc97xoydzxd7w9",
                output,
                quiet=False,
            )

        # Check if deepspeed is installed and initialized
        try:
            from deepspeed.comm.comm import is_initialized

            # Test if deepspeed is initialized
            if not is_initialized():
                raise Exception("Deepspeed is not initialized.")

            deepspeedEnabled = True
        except:
            deepspeedEnabled = False

        self.encoder = encode(output, verbose=False, deepspeed=deepspeedEnabled)
        self.labeler = label(output, verbose=False, deepspeed=deepspeedEnabled)

    def __len__(self):
        total_len = 0
        for task in self.nameToTask:
            if self.tasksReady[task]["ready"]:
                total_len += len(self.nameToTask[task])

        return total_len

    def get_dataloader(self, dataset, params: EvalParams = None):
        if params is None:
            params = self.evalParams
        if params.dataloader_fn is not None:
            return params.dataloader_fn(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            collate_fn=lambda x: x,
        )

        return dataloader
