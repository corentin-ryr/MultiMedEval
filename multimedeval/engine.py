"""The engine class."""

import getpass
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Type, Union

import gdown
import nltk

from torch.utils.data import DataLoader


from multimedeval.ct_rate import CTRATEReportGen, CTRATEClassification
from multimedeval.chestxray14 import ChestXray14
from multimedeval.chexbert.label import _encode, _label
from multimedeval.dynamic_datasets import find_datasets
from multimedeval.image_classification import (
    CBISDDSMCalcification,
    CBISDDSMMass,
    MIMICCXRImageClassification,
    PadUFES20,
    VinDrMammo,
)
from multimedeval.mednli import MedNLI
from multimedeval.mimic import MIMICCXRReportgen
from multimedeval.mimic_iii import MIMICIII
from multimedeval.mnist import (
    MNISTBlood,
    MNISTBreast,
    MNISTDerma,
    MNISTOct,
    MNISTOrganC,
    MNISTOrganS,
    MNISTPath,
    MNISTPneumonia,
    MNISTRetina,
    MNISTTissue,
)
from multimedeval.qa import MMLU, MedMCQA, MedQA, PubMedQA
from multimedeval.tqdm_loggable import TqdmLogging
from multimedeval.utils import (
    BatcherOutput,
    Benchmark,
    EvalParams,
    EvaluationOutput,
    SetupParams,
    file_writer_factory,
    BatcherInput,
)
from multimedeval.visualization import BenchmarkVisualizer
from multimedeval.vqa import SLAKE, DiffVQA, PathVQA, VQARad

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASKS: Set[Type[Benchmark]] = {
    MedQA,
    PubMedQA,
    MedMCQA,
    VQARad,
    PathVQA,
    DiffVQA,
    SLAKE,
    MIMICCXRReportgen,  # Setup not tested
    MIMICIII,
    MedNLI,
    MIMICCXRImageClassification,  # Setup not tested
    VinDrMammo,  # Setup not tested
    PadUFES20,
    CBISDDSMMass,
    CBISDDSMCalcification,
    MNISTOct,
    MNISTPath,
    MNISTBlood,
    MNISTBreast,
    MNISTDerma,
    MNISTOrganC,
    MNISTOrganS,
    MNISTPneumonia,
    MNISTRetina,
    MNISTTissue,
    MMLU,
    ChestXray14,
    CTRATEReportGen,
    CTRATEClassification,
    # # # "MNIST-OrganA": MNIST_OrganA,
    # # # "MNIST-Chest": MNIST_Chest,
}


TASKS_REQUIREMENTS: Dict[Type[Benchmark], List[str]] = {
    MIMICCXRReportgen: ["RadGraph", "Chexbert"],
    MIMICCXRImageClassification: ["Chexbert"],
    MIMICIII: ["RadGraph", "Chexbert"],
    DiffVQA: ["MIMIC-CXR Report Generation"],
}


class MultiMedEval:
    """The MultiMedEval engine."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the MultiMedEval engine.

        Args:
            logger: The logger to use for the evaluation info. Defaults to None.
        """
        self._config: Optional[SetupParams] = None
        self._physionet_username = None
        self._physionet_password = None
        self._hf_token = None

        if logger is None:
            self.logger = logging.getLogger("MultiMedEval")
        else:
            self.logger = logger

        self.eval_params = EvalParams()

        dynamic_datasets = find_datasets()
        print(f"Dynamic datasets: {dynamic_datasets}")
        TASKS.update(dynamic_datasets)
        print(f"TASKS: {TASKS}")

        self.tasks_ready = {}

        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)

        self.name_to_task: Dict[str, Benchmark] = {}
        self.name_to_requirements: Dict[str, List[str]] = {}
        for task_class in TASKS:
            benchmark: Benchmark = task_class(engine=self, logger=self.logger)
            self.name_to_task[benchmark.task_name] = benchmark
            self.name_to_requirements[benchmark.task_name] = (
                TASKS_REQUIREMENTS[task_class]
                if task_class in TASKS_REQUIREMENTS
                else []
            )

            self.tasks_ready[benchmark.task_name] = {
                "ready": False,
                "error": "Not setup yet",
            }

        print(f"Task names: {self.name_to_task}")

    def setup(self, setup_params: SetupParams, verbose: bool = True):
        """Setup the engine and all the tasks.

        Args:
            setup_params: The setup parameters.
            verbose: Whether or not to log during setup. Defaults to True.

        Returns:
            The tasks that are ready.
        """
        self.logger.info("Starting the setup of MultiMedEval.")
        self._config = setup_params
        tasks_to_skip: List[str] = []

        progress_bar = TqdmLogging(
            logger=self.logger, total=len(self.name_to_task) + 2, dynamic_ncols=True
        )
        progress_bar.set_description("Setup RadGraph")
        try:
            self._prepare_radgraph()
        except (ImportError, AttributeError) as e:
            self.tasks_ready["RadGraph"] = {"ready": False, "error": str(e)}
        else:
            self.tasks_ready["RadGraph"] = {"ready": True}
        progress_bar.update(1)

        progress_bar.set_description("Setup Chexbert")
        try:
            self._prepare_chexbert()
        except Exception as e:
            self.tasks_ready["Chexbert"] = {"ready": False, "error": str(e)}
        else:
            self.tasks_ready["Chexbert"] = {"ready": True}
        progress_bar.update(1)

        for task_name, task_item in self.name_to_task.items():
            progress_bar.set_description(f"Setup {task_name}")
            try:
                if task_name in tasks_to_skip:
                    raise ValueError(f"Task {task_name} is skipped")
                task_item.setup()
            except Exception as e:  # pylint: disable=broad-except
                self.tasks_ready[task_name] = {"ready": False, "error": str(e)}
            else:
                self.tasks_ready[task_name] = {"ready": True}

            progress_bar.update(1)
        progress_bar.close()

        final_message = "End of setup."

        if verbose:
            final_message += "\n"
            # Log a table of the tasks and their status
            final_message += "Task".ljust(35) + "Status".ljust(20) + "Error"
            for task_name, task_ready_item in self.tasks_ready.items():
                error = (
                    "No error."
                    if "error" not in task_ready_item
                    else task_ready_item["error"]
                )
                ready = "Ready" if task_ready_item["ready"] else "Problem"
                final_message += (
                    "\n" + task_name.ljust(35) + ready.ljust(20) + str(error)
                )

        self.logger.info(final_message)

        return self.tasks_ready

    def eval(
        self,
        tasks_to_evaluate: Union[str, List[str]],
        batcher: Callable,
        eval_params: Optional[EvalParams] = None,
    ):
        """Evaluate the tasks.

        Args:
            tasks_to_evaluate: Tasks on which to run the evaluation.
            batcher: The batcher to use.
            eval_params: The evaluation parameters. Defaults to None.

        Returns:
            The results of the evaluation.
        """
        if batcher is None:
            raise ValueError(
                "The engine was not initialized with a batcher, "
                "please provide a batcher to the engine"
            )

        self.eval_params = eval_params if eval_params is not None else self.eval_params

        if not os.path.exists(self.eval_params.run_name):
            os.mkdir(self.eval_params.run_name)

        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(tasks_to_evaluate, list):
            if len(tasks_to_evaluate) == 0:
                tasks_to_evaluate = list(self.name_to_task.keys())
            results = {}
            for x in tasks_to_evaluate:
                task_metrics = self.eval(x, batcher, eval_params)
                if task_metrics is None:
                    continue
                results[x] = task_metrics

                self._update_results_file(results)

            return results

        if tasks_to_evaluate not in self.name_to_task:
            self.logger.warning(
                "Task %s not in %s",
                tasks_to_evaluate,
                list(self.name_to_task.keys()),
            )
            return None

        # Check if the requirements are satisfied
        list_requirements = [tasks_to_evaluate] + self.name_to_requirements[
            tasks_to_evaluate
        ]
        for req in list_requirements:
            if not self.tasks_ready[req]["ready"]:
                error = (
                    self.tasks_ready[req]["error"]
                    if "error" in self.tasks_ready[req]
                    else "No error message"
                )

                warning_message = (
                    f"Task {tasks_to_evaluate} requires {req} to be ready: {error}"
                )
                self.logger.warning(warning_message)
                return None

        evaluation: Benchmark = self.name_to_task[tasks_to_evaluate]

        predictions = self._run_inference(evaluation, batcher)
        task_result: EvaluationOutput = evaluation.evaluate(predictions)

        if task_result.answer_log is not None:
            file_writer_factory("csv")(
                task_result.answer_log,
                f"{self.eval_params.run_name}/{evaluation.task_name}",
            )

        self.logger.info("Done task %s", str(tasks_to_evaluate))

        return task_result.metrics

    def _update_results_file(self, results):
        try:
            with open(
                f"{self.eval_params.run_name}/results.json", "r", encoding="utf-8"
            ) as f:
                metrics = json.load(f)
        except IOError:
            metrics = {}

        metrics.update(results)
        file_writer_factory("json")(metrics, f"{self.eval_params.run_name}/results")

    def _run_inference(self, task: Benchmark, batcher) -> List[Dict[str, Union[int, BatcherOutput]]]:
        info_message = (
            f"======================== Running inference on {task.task_name} "
            "========================"
        )
        self.logger.info(info_message)

        dataloader = self.get_dataloader(task)
        kwargs_format_question = (
            {
                "include_indication": self.eval_params.mimic_cxr_include_indication_section
            }
            if task.task_name == "MIMIC-CXR Report Generation"
            else {}
        )

        predictions = []
        for batch in TqdmLogging(self.logger, dataloader, desc="Running inference"):
            batch_prompts = []
            for el in batch:
                sample = el["sample"]
                batcher_input = task.format_question(sample, **kwargs_format_question)
                if self.eval_params.fewshot and task.get_prompt() is not None:
                    few_shot_input = task.get_prompt()
                    few_shot_input_plus_one = few_shot_input + batcher_input
                    batch_prompts.append(few_shot_input_plus_one)
                else:
                    batch_prompts.append(batcher_input)

            answers: List[BatcherOutput] = batcher(batch_prompts)

            # Validate that the answers are the same length as the batch and that they are instances of BatcherOutput
            if len(answers) != len(batch) or not all(
                isinstance(answer, BatcherOutput) for answer in answers
            ):
                raise ValueError(
                    "The batcher should return a list of BatcherOutput instances with the same length as the input batch."
                )

            for el, answer in zip(batch, answers):
                predictions.append({"idx": el["idx"], "answer": answer})

        return predictions

    def visualization(self):
        """Generate visualizations for the tasks."""
        benchmarks = [
            self.name_to_task[x]
            for x, task_item in self.tasks_ready.items()
            if (task_item["ready"] and x in self.name_to_task)
        ]
        print(benchmarks)
        visualizer = BenchmarkVisualizer(benchmarks)
        visualizer.sunburst_modalities()
        visualizer.sunburst_tasks()
        visualizer.table_image_classification()
        visualizer.sankey_diagram()
        # visualizer.sankeyD3Blocks()

    def get_physionet_credentials(self):
        """Returns the PhysioNet credentials.

        Returns:
            A tuple with the PhysioNet username and password.
        """
        if self._physionet_password is None or self._physionet_username is None:
            self._physionet_username = self.get_config()["physionet_username"]
            self._physionet_password = self.get_config()["physionet_password"]
            if not self._physionet_username or not self._physionet_password:
                self.logger.info(
                    "To setup the tasks that use a PhysioNet dataset, the scripts "
                    "requires the PhysioNet username and password."
                )
                self._physionet_username = input("Enter your username: ")
                self._physionet_password = getpass.getpass("Enter your password: ")

        return self._physionet_username, self._physionet_password
    
    def get_huggingface_token(self):
        if self._hf_token is None:
            self._hf_token = self.get_config()["hf_token"]
            if not self._hf_token:
                if 'HF_TOKEN' in os.environ and os.environ['HF_TOKEN']:
                    self._hf_token = os.environ['HF_TOKEN']
                else:
                    self.logger.info(
                    "To setup the tasks that use a protected hugggingface dataset, the scripts "
                    "requires the personal hugging face token."
                )
                self._hf_token = input("Enter your personal huggingface_token: ")
        return self._hf_token
        

    def get_huggingface_token(self):
        if self._hf_token is None:
            self._hf_token = self.get_config()["hf_token"]
            if not self._hf_token:
                if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
                    self._hf_token = os.environ["HF_TOKEN"]
                else:
                    self.logger.info(
                        "To setup the tasks that use a protected hugggingface dataset, the scripts "
                        "requires the personal hugging face token."
                    )
                self._hf_token = input("Enter your personal huggingface_token: ")
        return self._hf_token

    def get_config(self) -> dict:
        """Get the evaluation parameters as a dictionary.

        Returns:
            The eval parameters.
        """
        if self._config is None:
            raise ValueError(
                "The engine was not setup, please run the setup method first."
            )

        return asdict(self._config)

    def _write_to_tensorboard(self, results):
        writer = self.eval_params.tensorboard_writer
        if writer is None:
            return

        run_name = self.eval_params.run_name
        task_name = results["name"]

        metrics = results["value"]
        for metric in metrics:
            metric_value = metrics[metric]
            writer.add_scalar(
                f"{run_name}/{task_name}/{metric}",
                metric_value,
                global_step=self.eval_params.tensorboard_step,
            )

    def _prepare_radgraph(self):
        from radgraph import F1RadGraph

        # Check if deepspeed is installed and initialized
        try:
            from deepspeed.comm.comm import (  # noqa # pylint: disable=import-outside-toplevel  # type: ignore
                is_initialized,
            )

            # Test if deepspeed is initialized
            if not is_initialized():
                raise ImportError("Deepspeed is not initialized.")
        except ImportError:
            pass
        else:
            raise ImportError("Deepspeed is initialized.")

        device = -1 if self.get_config()["device"] != "cuda" else 0
        self.radgraph = F1RadGraph(
            reward_level="partial", cuda=device, model_type="radgraph"
        )

    def _prepare_chexbert(self):
        # Download the Chexbert checkpoint from
        # https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
        path = self.get_config()["chexbert_dir"]

        if path is None:
            raise ValueError("chexbert_dir is not set in the config file.")

        output = os.path.join(path, "chexbert.pth")

        if not os.path.exists(output):
            os.makedirs(path, exist_ok=True)
            gdown.download(
                "https://stanfordmedicine.app.box.com/shared/static/"
                "c3stck6w6dol3h36grdc97xoydzxd7w9",
                output,
                quiet=False,
            )

        # Check if deepspeed is installed and initialized
        try:
            from deepspeed.comm.comm import (  # noqa # pylint: disable=import-outside-toplevel # type: ignore
                is_initialized,
            )

            # Test if deepspeed is initialized
            if not is_initialized():
                raise ImportError("Deepspeed is not initialized.")

            deepspeed_enabled = True
        except ImportError:
            deepspeed_enabled = False

        self.encoder = _encode(output, verbose=False, deepspeed=deepspeed_enabled)
        self.labeler = _label(output, verbose=False, deepspeed=deepspeed_enabled)

    def __len__(self):
        """The total number of samples in the tasks."""
        total_len = 0
        for task, task_item in self.name_to_task.items():
            if self.tasks_ready[task]["ready"]:
                total_len += len(task_item)

        return total_len

    def get_dataloader(self, dataset, params: Optional[EvalParams] = None):
        """Return a DataLoader for the dataset.

        Args:
            dataset: The dataset to prepare.
            params: The parameters to create the dataloader. Defaults to None.

        Returns:
            A dataloader.
        """
        if params is None:
            params = self.eval_params
        if params.dataloader_fn is not None:
            return params.dataloader_fn(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            collate_fn=lambda x: x,
        )

        return dataloader
