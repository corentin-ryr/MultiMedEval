from multimedeval import MultiMedEval, SetupParams, EvalParams
import json
import os
import pytest


def batcher(prompts):
    return ["Dummy answer" for _ in range(len(prompts))]

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_image_classification():
    engine = MultiMedEval()
    config = json.load(open("MedMD_config.json"))
    engine.setup(SetupParams(MNIST_Oct_dir=config["MNIST_Oct_dir"]))
    
    
    engine.eval(["OCTMNIST"], batcher, EvalParams())