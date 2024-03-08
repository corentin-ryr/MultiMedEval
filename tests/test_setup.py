from multimedeval import EvalParams, MultiMedEval, SetupParams
import json
import logging
import shutil
import os
import pytest

logging.basicConfig(level=logging.INFO)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_setup():

    path = "/home/croyer/scratch/testSetupData"
    # Check if folder already exists
    if os.path.exists(path):
        raise Exception(
            "This test will create a folder called testSetupData and then delete it. A folder with the same name already exists. Abording test to avoid data loss."
        )

    engine = MultiMedEval()

    config = json.load(open("MedMD_config.json"))

    setupParams = SetupParams(
        MedQA_dir=path,
        PubMedQA_dir=path,
        MedMCQA_dir=path,
        VQA_RAD_dir=path,
        Path_VQA_dir=path,
        SLAKE_dir=path,
        CBIS_DDSM_dir=path,
        MedNLI_dir=path,
        CheXBert_dir=path,
        physionet_password=config["physionet_password"],
        physionet_username=config["physionet_username"],
    )

    tasksReady = engine.setup(setupParams)


    json.dump(tasksReady, open("tasksReady.json", "w"))

    shutil.rmtree(path)

    tasksToCheck = [
        "MedQA",
        "PubMedQA",
        "MedMCQA",
        "VQA-Rad",
        "VQA-Path",
        "SLAKE",
        "CBIS-DDSM Mass",
        "CBIS-DDSM Calcification",
        "MedNLI",
        "Chexbert",
        "RadGraph",
    ]
    for task in tasksToCheck:
        if task in tasksReady:
            assert tasksReady[task]["ready"]
        else:
            assert False
