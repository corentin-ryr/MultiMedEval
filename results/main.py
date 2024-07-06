"""Main script to run the evaluation."""

import argparse
import json
import logging
from abc import abstractmethod

from llava_med.batcher import batcherLLaVA_Med
from LLMs.batcher import batcherLlama, batcherMedAlpaca, batcherMistral, batcherPMCLlama
from rad_fm.batcher import RadFMBatcher

from multimedeval import EvalParams, MultiMedEval, SetupParams

logging.basicConfig(level=logging.INFO)

BATCHERS = {
    "RadFM": RadFMBatcher,
    "Llama": batcherLlama,
    "MedAlpaca": batcherMedAlpaca,
    "Mistral": batcherMistral,
    "PMCLlama": batcherPMCLlama,
    "LLaVA_Med": batcherLLaVA_Med,
}


class Batcher:
    """Batcher abstract class."""

    @abstractmethod
    def __init__(self, model_path):
        """Initialize the batcher."""
        pass


def main(batcherName):
    """Main function."""
    # The two path are PATH/TO/MODEL/RadFM/Language_files and PATH/TO/MODEL/RadFM_cleaned/Language_files
    batcher = RadFMBatcher(**json.load(open("configPaths.json")))

    mmb = MultiMedEval()

    setupParams = SetupParams(**json.load(open("MedMD_config.json")))
    mmb.setup(setupParams)

    mmb.eval(
        [],
        batcher,
        EvalParams(
            batch_size=32,
            run_name=f"results_{batcherName}",
            fewshot=False,
            mimic_cxr_include_indication_section=True,
        ),
    )

    logging.info("Done")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--batcher", type=str, required=True)

    args = argParser.parse_args()
    batcherName = args.batcher

    if batcherName not in BATCHERS:
        raise Exception(
            f"Batcher {batcherName} not found. Available batchers: {list(BATCHERS.keys())}"
        )

    main(batcherName)
