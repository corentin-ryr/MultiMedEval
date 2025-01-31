"""Main script to run the evaluation."""

import argparse
import json
import logging

# from results.llava_med.batcher import batcherLLaVA_Med
# from results.llms.batcher import batcherLlama, batcherMedAlpaca, batcherMistral, batcherPMCLlama
# from results.rad_fm.batcher import RadFMBatcher
from results.ct_clip.ct_clip import BatcherCTClip

from multimedeval import EvalParams, MultiMedEval, SetupParams

logging.basicConfig(level=logging.INFO)

BATCHERS = {
    # "RadFM": RadFMBatcher,
    # "Llama": batcherLlama,
    # "MedAlpaca": batcherMedAlpaca,
    # "Mistral": batcherMistral,
    # "PMCLlama": batcherPMCLlama,
    # "LLaVA-Med": batcherLLaVA_Med,
    "CT-CLIP": BatcherCTClip,
}


def main(batcherName):
    """Main function."""
    # The two path are PATH/TO/MODEL/RadFM/Language_files and PATH/TO/MODEL/RadFM_cleaned/Language_files
    batcher = BATCHERS[batcherName](**json.load(open("results_config.json")))
    # RadFMBatcher(**json.load(open("configPaths.json")))

    mmb = MultiMedEval()

    setupParams = SetupParams(**json.load(open("MedMD_config.json")))
    mmb.setup(setupParams)

    eval_output = mmb.eval(
        ["CT-RATE Image Classification"],
        batcher,
        EvalParams(
            batch_size=10,
            run_name=f"results_{batcherName}",
            fewshot=False,
            mimic_cxr_include_indication_section=True,
            num_workers=48,
        ),
    )

    logging.info(f"Done: {eval_output}")


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
