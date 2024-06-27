import json
import os

import pytest

from multimedeval import EvalParams, MultiMedEval, SetupParams

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.parametrize(
    "batcherAnswer, expectedAccuracy",
    [
        ("entailment", 474 / 1422),
        ("contradiction", 474 / 1422),
        ("neutral", 474 / 1422),
        ("normal", 0.0),
    ],
)
def test_mednli(batcherAnswer, expectedAccuracy):

    def batcher(prompts):
        return [batcherAnswer for _ in range(len(prompts))]

    engine = MultiMedEval()
    config = (
        json.load(open("tests/test_config.json"))
        if IN_GITHUB_ACTIONS
        else json.load(open("MedMD_config.json"))
    )
    if IN_GITHUB_ACTIONS:
        config["physionet_username"] = os.getenv("PHYSIONET_USERNAME")
        config["physionet_password"] = os.getenv("PHYSIONET_PASSWORD")

    try:
        engine.setup(
            SetupParams(
                MedNLI_dir=config["MedNLI_dir"],
                physionet_username=config["physionet_username"],
                physionet_password=config["physionet_password"],
            )
        )
    except:
        assert False

    results = engine.eval(["MedNLI"], batcher, EvalParams())

    print(results)

    if "MedNLI" not in results:
        assert False

    assert (results["MedNLI"]["accuracy"] - expectedAccuracy) < 0.01
