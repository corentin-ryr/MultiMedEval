from multimedeval import EvalParams, MultiMedEval, SetupParams
import json
import logging
import pytest
import os

logging.basicConfig(level=logging.INFO)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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
            json.load(open("tests/test_config.json")) if IN_GITHUB_ACTIONS else json.load(open("MedMD_config.json"))
        )
        tasksToPrepare = [
            "MedQA",
            "PubMedQA",
            "MedMCQA",
            "VQA-Rad",
            "VQA-Path",
            "SLAKE",
            "MedNLI",
            "OCTMNIST",
            "Pad UFES 20",
        ]

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
    def test_running_qa(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        tasks = ["MedQA", "PubMedQA", "MedMCQA"]

        try:
            results = self.engine.eval(tasks, batcher, evalParams=evalParams)
        except:
            assert False

        for task in tasks:
            assert task in results

    @pytest.mark.order(4)
    def test_running_vqa(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        tasks = ["VQA-Rad", "VQA-Path", "SLAKE"]
        try:
            results = self.engine.eval(tasks, batcher, evalParams=evalParams)
        except:
            assert False

        for task in tasks:
            assert task in results

    @pytest.mark.order(5)
    def test_running_nli(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        tasks = ["MedNLI"]
        try:
            results = self.engine.eval(tasks, batcher, evalParams=evalParams)
        except:
            assert False

        for task in tasks:
            assert task in results

    # @pytest.mark.order(6)
    # def test_running_reportcomparison(self):
    #     evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

    #     tasks = ["MIMIC-III"]
    #     try:
    #         results = self.engine.eval(tasks, batcher, evalParams=evalParams)
    #     except:
    #         assert False

    #     for task in tasks:
    #         assert task in results

    # @pytest.mark.order(7)
    # @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
    # def test_running_image_classification(self):
    #     evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

    #     tasks = [
    #         "MIMIC-CXR Image Classficication",
    #         # "VinDr Mammo",
    #         "Pad UFES 20",
    #         # "CBIS-DDSM Mass",
    #         # "CBIS-DDSM Calcification",
    #         "OCTMNIST",
    #         # "PathMNIST",
    #         # "PneumoniaMNIST",
    #         # "RetinaMNIST",
    #         # "BloodMNIST" "OrganCMNIST",
    #         # "DermaMNIST",
    #         # "BreastMNIST",
    #         # "TissueMNIST",
    #         # "OrganSMNIST",
    #     ]
    #     try:
    #         results = self.engine.eval(
    #             tasks,
    #             batcher,
    #             evalParams=evalParams,
    #         )
    #     except:
    #         assert False

    #     for task in tasks:
    #         assert task in results
