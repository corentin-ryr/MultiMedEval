from multimedeval import EvalParams, MultiMedEval, SetupParams
import json
import logging
import pytest
import shutil
import os

logging.basicConfig(level=logging.INFO)


def test_setup():

    path = "/home/croyer/scratch/testSetupData"
    # Check if folder already exists
    if os.path.exists(path):
        raise Exception(
            "This test will create a folder called testSetupData and then delete it. A folder with the same name already exists. Abording test to avoid data loss."
        )

    engine = MultiMedEval()

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
    )

    tasksReady = engine.setup(setupParams)

    for task in tasksReady:
        if task in [
            "MedQA",
            "PubMedQA",
            "MedMCQA",
            "VQA-RAD",
            "VQA-Path",
            "SLAKE",
            "CBIS-DDSM Mass",
            "CBIS-DDSM Calcification",
            "MedNLI",
            "Chexbert",
            "RadGraph",
        ]:
            assert tasksReady[task]["ready"] == True

    # Delete the setup data folder
    shutil.rmtree(path)
