"""Module for the task families."""

import os
from abc import abstractmethod
from typing import List, Optional, Union, Literal
import nltk
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from torchmetrics import AUROC, Accuracy, F1Score
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics.segmentation import GeneralizedDiceScore, HausdorffDistance, MeanIoU

from multimedeval.report_comparison_utils import (
    compute_bertscore,
    compute_composite,
    compute_meteor,
)
from multimedeval.utils import (
    Benchmark,
    EvaluationOutput,
    _exact_entity_token_if_rel_exists_reward,
    clean_str,
)


class QA(Benchmark):
    """A benchmark for Question Answering tasks."""

    def __init__(self, **kwargs):
        """Initialize the QA task."""
        super().__init__(**kwargs)
        self.task = "QA"

    @abstractmethod
    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Returns the raw answer. Defaults to False.
        """

    @abstractmethod
    def get_predicted_answer(self, pred: str, sample):
        """Converts the free form text output to the answer index.

        Args:
            pred: The free form text output of the model.
            sample: The sample used to generate the answer.
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        correct_answers = 0
        total_answers = 0

        answers_log = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            gold = self.get_correct_answer(sample)
            pred = self.get_predicted_answer(answer, sample)
            if pred == gold:
                correct_answers += 1
            total_answers += 1

            answers_log.append(
                (
                    self.get_correct_answer(sample, full_text=True),
                    answer,
                    pred,
                    gold,
                    pred == gold,
                )
            )

        metrics = {"accuracy": correct_answers / total_answers}

        return EvaluationOutput(answer_log=answers_log, metrics=metrics)


class VQA(Benchmark):
    """A benchmark for Visual Question Answering tasks."""

    def __init__(self, **kwargs):
        """Initialize the VQA task."""
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "VQA"
        self.wnl = WordNetLemmatizer()

    def _preprocess(self, text):
        tokenized_text = set(text.split())
        tokenized_text.discard("")
        return tokenized_text

    @abstractmethod
    def get_correct_answer(self, sample):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        answers_log = [
            [
                "correct",
                "predicted",
                "F1",
                "BLEU",
                "recall",
                "correct tokens",
                "predicted tokens",
            ]
        ]
        bleu_scores = []
        f1 = []
        recall = []
        closed_questions = []
        closed_questions_correct = 0

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            # Compute the number of tokens recalled in the answer
            clean_correct = clean_str(self.get_correct_answer(sample))
            clean_predicted = clean_str(answer)

            predicted_tokens = self._preprocess(clean_predicted)
            correct_tokens = self._preprocess(clean_correct)

            current_precision = (
                len(predicted_tokens.intersection(correct_tokens))
                / len(predicted_tokens)
                if len(predicted_tokens) > 0
                else 0
            )
            current_recall = (
                len(predicted_tokens.intersection(correct_tokens)) / len(correct_tokens)
                if len(correct_tokens) > 0
                else 0
            )
            current_f1 = (
                2
                * (current_precision * current_recall)
                / (current_precision + current_recall + 1e-8)
            )
            f1.append(current_f1)
            recall.append(current_recall)

            current_bleu = self.bleu([clean_predicted], [[clean_correct]]).item()
            bleu_scores.append(current_bleu)

            # If the question is closed, decide if it is correct or not
            if clean_correct in ["yes", "no"]:
                closed_questions.append(True)

                if correct_tokens == {"no"}:
                    correct_tokens.add("not")

                closed_q_recall = (
                    len(predicted_tokens.intersection(correct_tokens))
                    / len(correct_tokens)
                    if len(correct_tokens) > 0
                    else 0
                )
                if closed_q_recall >= 0.4:
                    closed_questions_correct += 1
            else:
                closed_questions.append(False)

            answers_log.append(
                (
                    self.get_correct_answer(sample),
                    answer,
                    current_f1,
                    current_bleu,
                    current_recall,
                    correct_tokens,
                    predicted_tokens,
                )
            )

        # Compute the accuracy for closed questions
        open_questions_recall = []
        open_questions_accuracy = 0
        for idx, is_closed in enumerate(closed_questions):
            if not is_closed:
                open_questions_recall.append(recall[idx])
                if recall[idx] >= 0.75:
                    open_questions_accuracy += 1

        metrics = {
            "bleu": sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) > 0 else 0,
            "F1": sum(f1) / len(f1) if len(f1) > 0 else 0,
            "recall": sum(recall) / len(recall) if len(recall) > 0 else 0,
            "closedQuestionsAccuracy": (
                closed_questions_correct / sum(closed_questions)
                if sum(closed_questions) > 0
                else 0
            ),
            "openQuestionsRecall": (
                sum(open_questions_recall) / len(open_questions_recall)
                if len(open_questions_recall) > 0
                else 0
            ),
            "openQuestionsAccuracy": (
                open_questions_accuracy / len(open_questions_recall)
                if len(open_questions_recall) > 0
                else 0
            ),
        }

        return EvaluationOutput(answer_log=answers_log, metrics=metrics)


class ImageClassification(Benchmark):
    """A benchmark for image classification tasks."""

    def __init__(self, **kwargs) -> None:
        """Initialize the ImageClassification task."""
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "Image Classification"
        self.scoring_type = "multiclass"
        self.num_classes = None
        self.options: List[str] = []
        self.path: Optional[os.PathLike] = None

    @abstractmethod
    def get_correct_answer(self, sample, full_text=False) -> Union[int, str, List[int]]:
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Whether or not to return the raw answer. Defaults to False.

        Returns:
            The correct answer.
        """

    @abstractmethod
    def format_question(self, sample, prompt=False):
        """Format the question for the model.

        Args:
            sample: The sample to format.
            prompt: Whether or not to have the answer in the fromatted question. Defaults to False.
        """

    @abstractmethod
    def get_predicted_answer(self, answer) -> Union[int, List[int]]:
        """Converts the free form text output to the answer index.

        Args:
            sample (_type_): The sample used to generate the answer
            answer (_type_): The free form text output of the model

        Returns:
            int: The index of the answer
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        scorer_args = {"task": self.scoring_type, "average": "macro"}
        scorer_args.update(
            {"num_classes": self.num_classes}
            if self.scoring_type == "multiclass"
            else {"num_labels": self.num_classes}
        )
        f1_scorer = F1Score(**scorer_args)
        auroc_scorer = AUROC(**scorer_args)
        accuracy = Accuracy(**scorer_args)

        predicted_answers = []
        ground_truth = []
        answers_log = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            pred = self.get_predicted_answer(answer)
            gt = self.get_correct_answer(sample)

            predicted_answers.append(pred)
            ground_truth.append(gt)

            answers_log.append(
                (
                    f"{self.get_correct_answer(sample, full_text=True)} (index {gt})",
                    f"{answer} (index {pred})",
                    gt == pred,
                )
            )

        # Convert pred and gt to tensor
        predicted_answers = torch.tensor(predicted_answers)
        if self.scoring_type == "multiclass":
            predicted_answers = torch.nn.functional.one_hot(
                predicted_answers, num_classes=self.num_classes
            )  # pylint: disable=not-callable
        predicted_answers = predicted_answers.to(torch.float32)
        ground_truth = torch.tensor(ground_truth)

        f1_macro = f1_scorer(predicted_answers, ground_truth).item()
        auroc = auroc_scorer(predicted_answers, ground_truth).item()
        acc = accuracy(predicted_answers, ground_truth).item()

        metrics = {"AUC-macro": auroc, "F1-macro": f1_macro, "Accuracy": acc}

        return EvaluationOutput(answer_log=answers_log, metrics=metrics)

class Segmentation(Benchmark):
    """A benchmark for segmentation tasks."""

    def __init__(self, **kwargs) -> None:
        """Initialize the Segmentation task."""
        super().__init__(**kwargs)

        self.seg_type: Optional[Literal["seg_mask", "bbox"]] = "seg_mask"
        self.task = "Segmentation"
        self.num_classes = 1
        self.path: Optional[os.PathLike] = None

    @abstractmethod
    def get_correct_answer(self, sample):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.

        Returns:
            The correct segmentation mask.
        """

    @abstractmethod
    def format_question(self, sample, prompt=False):
        """Format the question for the model.

        Args:
            sample: The sample to format.
            prompt: Whether or not to have the answer in the fromatted question. Defaults to False.
        """

    @abstractmethod
    def get_predicted_answer(self, answer):
        """Converts the free form text output to the answer index.

        Args:
            sample (_type_): The sample used to generate the answer
            answer (_type_): The predicted segmentation mask

        Returns:
            int: The index of the answer
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        predicted_answers = []
        ground_truth = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            pred = self.get_predicted_answer(answer)
            gt = self.get_correct_answer(sample)

            predicted_answers.append(pred)
            ground_truth.append(gt)

        if self.seg_type == "seg_mask":
            scorer_args = {"num_classes": self.num_classes, "per_class" :False, "include_background": False}

            dice_scorer = GeneralizedDiceScore(**scorer_args)

            predicted_answers = torch.tensor(predicted_answers)
            predicted_answers = predicted_answers.to(torch.float32)
            ground_truth = torch.tensor(ground_truth)

            dice = dice_scorer(predicted_answers, ground_truth).item()

            metrics = {"dice": dice}
        elif self.seg_type == "bbox":
            preds = [
            dict(
                boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=torch.tensor([0.536]),
                labels=torch.tensor([0]),
            )
            ]
            target = [
            dict(
                boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                labels=torch.tensor([0]),
            )
            ]
            # metric = MeanAveragePrecision()
            # metric.update(preds, target)
            metric = 1

            # print(metric.compute())
            metrics = {"mAP": metric}

        return EvaluationOutput(metrics=metrics)

class ReportComparison(Benchmark):
    """A benchmark for report comparison tasks."""

    def __init__(self, **kwargs) -> None:
        """Initialize the ReportComparison task."""
        super().__init__(**kwargs)
        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2, weights=[1 / 2, 1 / 2])
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rouge_l = ROUGEScore(rouge_keys=("rougeL", "rouge1"))

        nltk.download("punkt_tab")

    def _compute_chexbert(self, hyp_reports, ref_reports):
        df = pd.DataFrame(columns=["Report Impression"], data=ref_reports)
        labels_reference = self.engine.encoder(df)

        df = pd.DataFrame(columns=["Report Impression"], data=hyp_reports)
        labels_hypothesis = self.engine.encoder(df)

        # Compute the vector similarity between the reference and the geenrated reports
        return torch.cosine_similarity(labels_reference, labels_hypothesis)

    def _compute_radgraph(self, hyp_reports, ref_reports):

        f1_radgraph = []
        for hyp, ref in zip(hyp_reports, ref_reports):
            # Compute the F1-radgraph score
            (_, _, hyp_annotation_lists, ref_annotation_lists) = self.engine.radgraph(
                refs=[ref], hyps=[hyp]
            )
            f1_radgraph.append(
                _exact_entity_token_if_rel_exists_reward(
                    hyp_annotation_lists[0], ref_annotation_lists[0]
                )
            )
        return torch.tensor(f1_radgraph)

    def _evaluate_reports(self, hyp_reports, ref_reports):
        bleu1_scores = []
        bleu2_scores = []
        bleu4_scores = []
        rougel_scores = []
        rouge1_scores = []

        for hyp, ref in zip(hyp_reports, ref_reports):
            bleu1_scores.append(self.bleu_1([hyp], [[ref]]).item())
            bleu2_scores.append(self.bleu_2([hyp], [[ref]]).item())
            bleu4_scores.append(self.bleu_4([hyp], [[ref]]).item())
            current_rouge = self.rouge_l([hyp], [[ref]])
            rougel_scores.append(current_rouge["rougeL_fmeasure"].item())
            rouge1_scores.append(current_rouge["rouge1_fmeasure"].item())

        f1_bertscore = compute_bertscore(hyp_reports, ref_reports)
        f1_bertscore_unscaled = compute_bertscore(
            hyp_reports, ref_reports, rescale=False
        )

        chexbert_similarity = self._compute_chexbert(hyp_reports, ref_reports)

        f1_radgraph = self._compute_radgraph(hyp_reports, ref_reports)

        bleu_scores = torch.tensor(bleu2_scores)

        radcliq_v0_scores = compute_composite(
            bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph
        )

        meteor_scores = compute_meteor(hyp_reports, ref_reports)

        return (
            bleu1_scores,
            bleu2_scores,
            bleu4_scores,
            rougel_scores,
            rouge1_scores,
            f1_bertscore_unscaled.tolist(),
            chexbert_similarity.tolist(),
            f1_radgraph.tolist(),
            radcliq_v0_scores.tolist(),
            meteor_scores,
            f1_bertscore.tolist(),
        )

    @abstractmethod
    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            full_text: Returns the raw answer. Defaults to False.
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        ref_reports = []
        hyp_reports = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            ref_reports.append(self.get_correct_answer(sample))
            hyp_reports.append(answer)

        (
            bleu1_scores,
            _,
            bleu4_scores,
            rougel_scores,
            rouge1_scores,
            f1_bertscore_unscaled,
            chexbert_similarity,
            f1_radgraph,
            radcliq_v0_scores,
            meteor_scores,
            _,
        ) = self._evaluate_reports(hyp_reports, ref_reports)

        metrics = {
            "bleu1": sum(bleu1_scores) / len(bleu1_scores),
            "bleu4": sum(bleu4_scores) / len(bleu4_scores),
            "f1-radgraph": sum(f1_radgraph) / len(f1_radgraph),
            "CheXBert vector similarity": sum(chexbert_similarity)
            / len(chexbert_similarity),
            "f1-bertscore": sum(f1_bertscore_unscaled) / len(f1_bertscore_unscaled),
            "radcliq": sum(radcliq_v0_scores) / len(radcliq_v0_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
            "rougeL": sum(rougel_scores) / len(rougel_scores),
            "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        }

        answers_log = zip(
            ref_reports, hyp_reports, bleu1_scores, bleu4_scores, rougel_scores
        )
        # Add a header to the log
        answers_log = [("ref", "hyp", "bleu1", "bleu4", "rougeL")] + list(answers_log)

        return EvaluationOutput(answer_log=answers_log, metrics=metrics)
