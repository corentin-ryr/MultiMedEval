"""Test for the image classification tasks."""

import json
import os
from typing import List
import pytest
from multimedeval import EvalParams, MultiMedEval, SetupParams
from multimedeval.utils import BatcherInput, BatcherOutput

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestImageClassification:
    """Tests loading all tasks and running them."""

    # Create the engine once for all tests
    def setup_class(self):
        """Set up the test class."""
        self.engine = MultiMedEval()

        config_file_path = (
            "tests/test_config.json" if IN_GITHUB_ACTIONS else "MedMD_config.json"
        )
        with open(config_file_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        self.engine.setup(
            SetupParams(
                mnist_oct_dir=config["mnist_oct_dir"],
                chestxray14_dir=config["chestxray14_dir"] if "chestxray14_dir" in config else None,
            )
        )

    # @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    @pytest.mark.parametrize(
        "batcher_answer, expected_accuracy, expected_macro_f1, expected_macro_auc",
        [
            ("choroidal neovascularization", 0.25, 0.1, 0.5),
            ("diabetic macular edema", 0.25, 0.1, 0.5),
            ("drusen", 0.25, 0.1, 0.5),
            ("normal", 0.25, 0.1, 0.5),
        ],
    )
    def test_image_classification(
        self, batcher_answer, expected_accuracy, expected_macro_f1, expected_macro_auc
    ):
        """Tests the image classification task.

        Args:
            batcher_answer: Answers generated by the batcher.
            expected_accuracy: Expected accuracy.
            expected_macro_f1: Expected macro F1.
            expected_macro_auc: Expected macro AUC.
        """

        def batcher(prompts: List[BatcherInput]) -> List[BatcherOutput]:
            output = []
            for _ in prompts:
                output.append(BatcherOutput(batcher_answer))
            return output

        results = self.engine.eval(["OCTMNIST"], batcher, EvalParams())

        print(results)

        if "OCTMNIST" not in results:
            # Find the element in the list that has that "name" metrics_OCTMNIST
            raise AssertionError()

        assert (results["OCTMNIST"]["Accuracy-macro"] - expected_accuracy) < 0.01
        assert (results["OCTMNIST"]["F1-macro"] - expected_macro_f1) < 0.01
        # assert (results["OCTMNIST"]["AUROC-macro"] - expected_macro_auc) < 0.01

    @pytest.mark.skipif(
        IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions."
    )
    @pytest.mark.parametrize(
        "batcher_answer, expected_macro_f1, expected_macro_auc",
        [
            ("Atelectasis", 0.0162226352840662, 0.5),
            # ("Cardiomegaly", 0.005727143492549658, 0.5),
            # ("Effusion", 0.021994730457663536, 0.5),
            # ("Infiltration", 0.027536988258361816, 0.5),
            # ("Mass", 0.0091323247179389, 0.5),
            # ("Nodule", 0.008518209680914879, 0.5),
            # ("Pneumonia", 0.0030318424105644226, 0.5),
            # ("Pneumothorax", 0.01347136590629816, 0.5),
            # ("Consolidation", 0.009459184482693672, 0.5),
            # ("Edema", 0.0049825748428702354, 0.5),
            # ("Emphysema", 0.0058504571206867695, 0.5),
            # ("Fibrosis", 0.002387263346463442, 0.5),
            # ("Pleural_Thickening", 0.006106649991124868, 0.5),
            # ("Hernia", 0.00047837840975262225, 0.5),
        ],
    )
    def test_chestxray14(self, batcher_answer, expected_macro_f1, expected_macro_auc):
        """Tests the chestxray14 task.

        Args:
            batcher_answer: Answers generated by the batcher.
            expected_macro_f1: Expected macro F1.
            expected_macro_auc: Expected macro AUC.
        """

        def batcher(prompts: List[BatcherInput]) -> List[BatcherOutput]:
            output = []
            for _ in prompts:
                output.append(BatcherOutput(batcher_answer))
            return output

        results = self.engine.eval(["ChestXray14"], batcher, EvalParams())

        print(results)

        if "ChestXray14" not in results:
            # Find the element in the list that has that "name" metrics_OCTMNIST
            raise AssertionError()

        assert (results["ChestXray14"]["F1-macro"] - expected_macro_f1) < 0.01
        assert (results["ChestXray14"]["AUC-macro"] - expected_macro_auc) < 0.01
