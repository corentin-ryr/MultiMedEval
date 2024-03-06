from multimedeval import EvalParams, MultiMedEval, SetupParams
import json
import logging
import pytest

logging.basicConfig(level=logging.INFO)


def batcher(prompts):
    return ["Dummy answer" for _ in range(len(prompts))]


class TestLoadingAll:

    # Create the engine once for all tests
    def setup_class(self):
        self.engine = MultiMedEval()

    # Do this test first
    @pytest.mark.order(1)
    def test_loading_all(self):
        config = json.load(open("MedMD_config.json"))
        setupParams = SetupParams(**config)
        tasksReady = self.engine.setup(setupParams=setupParams)

        for task in tasksReady:
            assert tasksReady[task]["ready"] == True

        assert isinstance(len(self.engine), int)

    # Do this test second
    def test_running_qa(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        results = self.engine.eval(["MedQA", "PubMedQA", "MedMCQA"], batcher, evalParams=evalParams)

        print(results)

    def test_running_vqa(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        results = self.engine.eval(["VQA-Rad", "VQA-Path", "SLAKE"], batcher, evalParams=evalParams)

        print(results)

    def test_running_nli(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        results = self.engine.eval(["MedNLI"], batcher, evalParams=evalParams)

        print(results)

    def test_running_reportcomparison(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        results = self.engine.eval(["MIMIC-CXR Report Generation", "MIMIC-III"], batcher, evalParams=evalParams)

        print(results)

    def test_running_image_classification(self):
        evalParams = EvalParams(batch_size=128, fewshot=True, num_workers=0)

        results = self.engine.eval(
            [
                "MIMIC-CXR Image Classficication",
                "VinDr Mammo",
                "Pad UFES 20",
                "CBIS-DDSM Mass",
                "CBIS-DDSM Calcification",
                "OCTMNIST",
                "PathMNIST",
                "PneumoniaMNIST",
                "RetinaMNIST",
                "BloodMNIST" "OrganCMNIST",
                "DermaMNIST",
                "BreastMNIST",
                "TissueMNIST",
                "OrganSMNIST",
            ],
            batcher,
            evalParams=evalParams,
        )

        print(results)
