import os
import pickle

import dill
import numpy as np
import torch
from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """

    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate((norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


def compute_bertscore(hypReports, refReports, rescale=True):
    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=rescale,
        idf=True,
        idf_sents=hypReports,
    )

    P, R, f1_bertscore = scorer.score(hypReports, refReports)
    return f1_bertscore


def compute_meteor(hypReports, refReports):
    meteor_scores = []
    for ref, hyp in zip(refReports, hypReports):
        # Tokenize the reference and hypothesis
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)

        # Compute the meteor score
        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))

    return meteor_scores


def compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph):

    # Get the current path to the module
    module_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(module_path, "radcliq-v1.dill"), "rb") as f:
        composite_metric_v0_model: CompositeMetric = dill.load(f)

    # The column need to be in the order [bleu, bertscore, chexbert, radgraph]
    input_data = torch.stack(
        [f1_radgraph, f1_bertscore, chexbert_similarity, bleu_scores],
        dim=1,
    )
    return composite_metric_v0_model.predict(input_data)
