"""Tests the MedNLI task."""

import json
import os

import pytest

from multimedeval import EvalParams, MultiMedEval, SetupParams

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.parametrize(
    "batcher_answer, expected_accuracy",
    [
        ("entailment", 474 / 1422),
        ("contradiction", 474 / 1422),
        ("neutral", 474 / 1422),
        ("normal", 0.0),
    ],
)
def test_mednli(batcher_answer, expected_accuracy):
    """Tests the MedNLI task.

    Args:
        batcher_answer: The answer from the batcher.
        expected_accuracy: The expected accuracy.
    """

    def batcher(prompts):
        return [batcher_answer for _ in range(len(prompts))]

    engine = MultiMedEval()

    config_file_name = (
        "tests/test_config.json" if IN_GITHUB_ACTIONS else "MedMD_config.json"
    )
    with open(config_file_name, "r", encoding="utf-8") as file:
        config = json.load(file)
    if IN_GITHUB_ACTIONS:
        config["physionet_username"] = os.getenv("PHYSIONET_USERNAME")
        config["physionet_password"] = os.getenv("PHYSIONET_PASSWORD")

    try:
        engine.setup(
            SetupParams(
                mednli_dir=config["mednli_dir"],
                physionet_username=config["physionet_username"],
                physionet_password=config["physionet_password"],
            )
        )
    except Exception as e:
        raise AssertionError(f"Error in setup. {e}") from e

    results = engine.eval(["MedNLI"], batcher, EvalParams())

    print(results)

    if "MedNLI" not in results:
        raise AssertionError()

    assert (results["MedNLI"]["accuracy"] - expected_accuracy) < 0.01
