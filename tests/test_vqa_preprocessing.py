"""Tests the VQA-Rad preprocessing."""

import json
import logging
import os

import pytest

from multimedeval import MultiMedEval, SetupParams
from multimedeval.utils import clean_str
from multimedeval.vqa import VQARad

logging.basicConfig(level=logging.INFO)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestVQAPreprocessing:
    """Tests the VQA-Rad preprocessing."""

    def setup_class(self):
        """Set up the test class."""
        self.engine = MultiMedEval()

        config_file_name = (
            "tests/test_config.json" if IN_GITHUB_ACTIONS else "MedMD_config.json"
        )
        with open(config_file_name, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.engine.setup(SetupParams(vqa_rad_dir=config["vqa_rad_dir"]))
        self.vqarad: VQARad = self.engine.name_to_task["VQA-Rad"]

    @pytest.mark.parametrize(
        "text, expected_set",
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
                "The image includes a variety of diseases related to the respiratory "
                "system. Some of these diseases are pneumonia, chronic obstructive "
                "pulmonary disease (COPD), asthma, and lung cancer.",
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
                "No, this is not a study of the chest. It seems to be a study "
                "involving the brain, as it mentions MRI and diffusion-weighted imaging (DWI).",
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
    def test_vqa_preprocessing(self, text, expected_set):
        """Tests the VQA-Rad preprocessing."""
        text = clean_str(text)
        print(text)
        tokenized_text = self.vqarad._preprocess(text)

        assert tokenized_text == expected_set
