"""Module for comparing reports."""

import os

import dill
import torch
from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

from multimedeval.utils import CompositeMetric


def compute_bertscore(hyp_reports, ref_reports, rescale=True):
    """Computes the BERTScore F1 score for the given reports.

    Args:
        hypReports: The generated reports.
        refReports: The reference reports.
        rescale: Whether or not to rescale the bert score. Defaults to True.

    Returns:
        The BERTScore F1 scores.
    """
    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=rescale,
        idf=True,
        idf_sents=hyp_reports,
    )

    _, _, f1_bertscore = scorer.score(hyp_reports, ref_reports)
    return f1_bertscore


def compute_meteor(hyp_reports, ref_reports):
    """Computes the METEOR score for the given reports.

    Args:
        hypReports: The generated reports.
        refReports: The reference reports.

    Returns:
        The METEOR scores.
    """
    meteor_scores = []
    for ref, hyp in zip(ref_reports, hyp_reports):
        # Tokenize the reference and hypothesis
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)

        # Compute the meteor score
        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))

    return meteor_scores


def compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph):
    """Computes the composite metric score.

    Args:
        bleu_scores: BLEU scores.
        f1_bertscore: BERTScore F1 scores.
        chexbert_similarity: CheXBert similarity scores.
        f1_radgraph: RadGraph F1 scores.

    Returns:
        Composite metric score.
    """
    # Get the current path to the module
    module_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(module_path, "radcliq-v1.dill"), "rb") as f:
        unpickler = ModuleRedirectUnpickler(f)
        composite_metric_v0_model: CompositeMetric = unpickler.load()

    # The column need to be in the order [bleu, bertscore, chexbert, radgraph]
    input_data = torch.stack(
        [f1_radgraph, f1_bertscore, chexbert_similarity, bleu_scores],
        dim=1,
    )
    return composite_metric_v0_model.predict(input_data)


class ModuleRedirectUnpickler(dill.Unpickler):
    """Unpickler that redirects the module name to the new module name."""

    def find_class(self, module, name):
        """Redirects the module name to the new module name.

        Args:
            module: the module to redirect
            name: the name of the module

        Returns:
            the class
        """
        # Redirect the module name if it matches the old module name
        if module == "multimedeval.reportComparisonUtils":
            module = "multimedeval.utils"
        return super().find_class(module, name)
