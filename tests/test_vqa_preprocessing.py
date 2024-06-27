import json
import logging
import os

import pytest

from multimedeval import EvalParams, MultiMedEval, SetupParams
from multimedeval.utils import cleanStr
from multimedeval.vqa import VQA_RAD

logging.basicConfig(level=logging.INFO)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestVQAPreprocessing:

    def setup_class(self):
        self.engine = MultiMedEval()

        config = (
            json.load(open("tests/test_config.json"))
            if IN_GITHUB_ACTIONS
            else json.load(open("MedMD_config.json"))
        )

        self.engine.setup(SetupParams(VQA_RAD_dir=config["VQA_RAD_dir"]))
        self.vqarad: VQA_RAD = self.engine.nameToTask["VQA-Rad"]

    @pytest.mark.parametrize(
        "text, expectedSet",
        [
            (
                "are regions of the brain infarcted?",
                {"are", "regions", "of", "brain", "infarcted"},
            ),
            (
                "is this image in the transverse plane?",
                {"is", "this", "image", "in", "transverse", "plane"},
            ),
            ("mri-dwi", {"mri", "dwi"}),
            (
                "The image includes a variety of diseases related to the respiratory system. Some of these diseases are pneumonia, chronic obstructive pulmonary disease (COPD), asthma, and lung cancer.",
                {
                    "related",
                    "copd",
                    "image",
                    "system",
                    "are",
                    "cancer",
                    "to",
                    "pneumonia",
                    "variety",
                    "diseases",
                    "disease",
                    "pulmonary",
                    "of",
                    "obstructive",
                    "some",
                    "chronic",
                    "includes",
                    "respiratory",
                    "these",
                    "lung",
                    "asthma",
                    "and",
                },
            ),
            ("     ", set()),
            (
                "No, this is not a study of the chest. It seems to be a study involving the brain, as it mentions MRI and diffusion-weighted imaging (DWI).",
                {
                    "study",
                    "this",
                    "imaging",
                    "is",
                    "diffusion",
                    "be",
                    "chest",
                    "no",
                    "to",
                    "brain",
                    "mentions",
                    "mri",
                    "not",
                    "seems",
                    "of",
                    "involving",
                    "weighted",
                    "it",
                    "and",
                    "dwi",
                    "as",
                },
            ),
        ],
    )
    def test_vqa_preprocessing(self, text, expectedSet):
        text = cleanStr(text)
        print(text)
        tokenizedText = self.vqarad._preprocess(text)

        assert tokenizedText == expectedSet
