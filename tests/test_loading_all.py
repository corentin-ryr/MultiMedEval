"""Tests loading all tasks and running them."""

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
    """Dummy batcher for the tests."""
    return ["Dummy answer" for _ in range(len(prompts))]


class TestLoadingAll:
    """Tests loading all tasks and running them."""

    # Create the engine once for all tests
    def __init__(self):
        """Set up the test class."""
        self.engine = MultiMedEval()

    # Do this test first
    @pytest.mark.order(1)
    def test_loading_all(self):
        """Tests loading all tasks."""
        config_file_name = (
            "tests/test_config.json" if IN_GITHUB_ACTIONS else "MedMD_config.json"
        )
        with open(config_file_name, "r", encoding="utf-8") as file:
            config = json.load(file)
        tasks_to_prepare = TASKS

        if IN_GITHUB_ACTIONS:
            config["physionet_username"] = os.getenv("PHYSIONET_USERNAME")
            config["physionet_password"] = os.getenv("PHYSIONET_PASSWORD")

        setup_params = SetupParams(**config)
        tasks_ready = self.engine.setup(setup_params=setup_params)

        for task in tasks_to_prepare:
            if task not in tasks_ready:
                raise AssertionError()
            assert tasks_ready[task]["ready"] is True

        assert isinstance(len(self.engine), int)

    @pytest.mark.order(2)
    def test_visualization(self):
        """Tests the visualization."""
        self.engine.visualization()

    @pytest.mark.order(3)
    @pytest.mark.parametrize(
        "task",
        TASKS,
    )
    def test_running_task(self, task):
        """Tests running a task.

        Args:
            task : Name of the task.
        """
        eval_params = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        tasks = [task]

        try:
            results = self.engine.eval(tasks, batcher, eval_params=eval_params)
        except Exception as e:
            raise AssertionError(f"Error in task {task}. {e}") from e

        for current_task in tasks:
            assert current_task in results

        # Check that the results.json file contains the results
        assert os.path.exists(
            os.path.join(self.engine.eval_params.run_name, "results.json")
        )

        with open(
            os.path.join(self.engine.eval_params.run_name, "results.json"),
            "r",
            encoding="utf-8",
        ) as f:
            results = json.load(f)

        assert task in results
