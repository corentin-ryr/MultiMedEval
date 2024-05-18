from multimedeval import MultiMedEval, SetupParams, EvalParams
import json
import os
import pytest


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.parametrize(
    "batcherAnswer, expectedAccuracy, expectedMacroF1, expectedMacroAUC",
    [
        ("choroidal neovascularization", 0.25, 0.1, 0.5),
        ("diabetic macular edema", 0.25, 0.1, 0.5),
        ("drusen", 0.25, 0.1, 0.5),
        ("normal", 0.25, 0.1, 0.5),
    ],
)
def test_image_classification(batcherAnswer, expectedAccuracy, expectedMacroF1, expectedMacroAUC):

    def batcher(prompts):
        return [batcherAnswer for _ in range(len(prompts))]

    engine = MultiMedEval()
    config = json.load(open("tests/test_config.json")) if IN_GITHUB_ACTIONS else json.load(open("MedMD_config.json"))
    engine.setup(SetupParams(MNIST_Oct_dir=config["MNIST_Oct_dir"]))

    results = engine.eval(["OCTMNIST"], batcher, EvalParams())

    print(results)

    if "OCTMNIST" not in results:
        # Find the element in the list that has that "name" metrics_OCTMNIST
        assert False

    assert (results["OCTMNIST"]["Accuracy"] - expectedAccuracy) < 0.01
    assert (results["OCTMNIST"]["F1-macro"] - expectedMacroF1) < 0.01
    assert (results["OCTMNIST"]["AUC-macro"] - expectedMacroAUC) < 0.01
