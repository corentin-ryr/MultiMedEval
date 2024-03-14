from multimedeval import MultiMedEval, SetupParams, EvalParams
import json
import os
import pytest


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.order(0)
@pytest.mark.parametrize(
    "batcherAnswer, expectedAccuracy",
    [
        ("entailment", 474/ 1422),
        ("contradiction", 474 / 1422),
        ("neutral", 474 / 1422),
        ("normal", 0.),
    ],
)
def test_mednli(batcherAnswer, expectedAccuracy):

    def batcher(prompts):
        return [batcherAnswer for _ in range(len(prompts))]

    engine = MultiMedEval()
    config = json.load(open("tests/test_config.json")) if IN_GITHUB_ACTIONS else json.load(open("MedMD_config.json"))
    engine.setup(SetupParams(MedNLI_dir=config["MedNLI_dir"]))

    results = engine.eval(["MedNLI"], batcher, EvalParams())

    print(results)

    if "MedNLI" in results:
        # Find the element in the list that has that "name" metrics_OCTMNIST
        for element in results["MedNLI"]:
            if element["name"] == "metrics_MedNLI":
                results = element["value"]
                break
        else:
            assert False

    assert (results["accuracy"] - expectedAccuracy) < 0.01