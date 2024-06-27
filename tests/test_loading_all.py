import json
import logging
import os

import pytest

from multimedeval import EvalParams, MultiMedEval, SetupParams

logging.basicConfig(level=logging.INFO)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

TASKS = [
    "MedQA",
    "PubMedQA",
    "MedMCQA",
    "VQA-Rad",
    "VQA-Path",
    "SLAKE",
    "MedNLI",
    "Pad UFES 20",
    "OCTMNIST",
    "PathMNIST",
    "PneumoniaMNIST",
    "RetinaMNIST",
    "BloodMNIST",
    "OrganCMNIST",
    "DermaMNIST",
    "BreastMNIST",
    "TissueMNIST",
    "OrganSMNIST",
    "CBIS-DDSM Mass",
    "CBIS-DDSM Calcification",
    "exampleDatasetQA",
    "exampleDatasetVQA",
    "MMLU",
]


def batcher(prompts):
    return ["Dummy answer" for _ in range(len(prompts))]


class TestLoadingAll:

    # Create the engine once for all tests
    def setup_class(self):
        self.engine = MultiMedEval()

    # Do this test first
    @pytest.mark.order(1)
    def test_loading_all(self):
        config = (
            json.load(open("tests/test_config.json"))
            if IN_GITHUB_ACTIONS
            else json.load(open("MedMD_config.json"))
        )
        tasksToPrepare = TASKS

        if IN_GITHUB_ACTIONS:
            config["physionet_username"] = os.getenv("PHYSIONET_USERNAME")
            config["physionet_password"] = os.getenv("PHYSIONET_PASSWORD")

        setupParams = SetupParams(**config)
        tasksReady = self.engine.setup(setupParams=setupParams)

        for task in tasksToPrepare:
            if task not in tasksReady:
                assert False
            assert tasksReady[task]["ready"] == True

        assert isinstance(len(self.engine), int)

    @pytest.mark.order(2)
    def test_visualization(self):
        self.engine.visualization()

    @pytest.mark.order(3)
    @pytest.mark.parametrize(
        "task",
        TASKS,
    )
    def test_running_task(self, task):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        tasks = [task]

        try:
            results = self.engine.eval(tasks, batcher, evalParams=evalParams)
        except:
            assert False

        for task in tasks:
            assert task in results

        # Check that the results.json file contains the results
        assert os.path.exists(
            os.path.join(self.engine.evalParams.run_name, "results.json")
        )

        with open(os.path.join(self.engine.evalParams.run_name, "results.json")) as f:
            results = json.load(f)

        assert task in results
