from multimedeval.utils import Benchmark, EvalParams, cleanStr, exact_entity_token_if_rel_exists_reward
from torchmetrics.text import BLEUScore, ROUGEScore
from torch.utils.data import DataLoader
from multimedeval.tqdm_loggable import tqdm_logging
from abc import abstractmethod
from nltk.stem import WordNetLemmatizer
from torchmetrics import F1Score, AUROC, Accuracy
import torch
from bert_score import BERTScorer
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import dill
from nltk.translate.meteor_score import meteor_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


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


class QA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = "QA"

    def run(self, params: EvalParams, batcher):
        self.logger.info(f"***** Benchmarking : {self.taskName} *****")

        correct_answers = 0
        total_answers = 0

        answersLog = []

        # Run the batcher for all data split in chunks
        dataloader = DataLoader(
            self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x
        )
        for batch in tqdm_logging(self.logger, dataloader, desc="Running inference"):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                if params.fewshot and self.getPrompt() is not None:
                    batchPrompts.append((self.getPrompt()[0] + text, self.getPrompt()[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                gold = self.getCorrectAnswer(batch[idx])
                pred = self.getPredictedAnswer(answer, batch[idx])
                if pred == gold:
                    correct_answers += 1
                total_answers += 1

                answersLog.append((self.getCorrectAnswer(batch[idx], fullText=True), answer, pred, gold, pred == gold))

            # break

        # TODO: add others metrics such as AUC, F1...
        metrics = {"accuracy": correct_answers / total_answers}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    @abstractmethod
    def getCorrectAnswer(self, sample, fullText=False):
        pass

    @abstractmethod
    def getPredictedAnswer(self, pred: str, sample):
        pass


class VQA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "VQA"
        self.wnl = WordNetLemmatizer()

    def run(self, params: EvalParams, batcher):
        self.logger.info(f"***** Benchmarking : {self.taskName} *****")

        answersLog = [["correct", "predicted", "F1", "BLEU", "recall", "correct tokens", "predicted tokens"]]
        bleuScores = []
        f1 = []
        recall = []

        closedQuestions = []
        # Run the batcher for all data split in chunks
        dataloader = DataLoader(
            self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x
        )
        for batch in tqdm_logging(
            self.logger,
            dataloader,
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                if params.fewshot and self.getPrompt() is not None:
                    batchPrompts.append((self.getPrompt()[0] + text, self.getPrompt()[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            for idx, answer in enumerate(answers):
                # Compute the number of tokens recalled in the answer
                cleanCorrect = cleanStr(self.getCorrectAnswer(batch[idx]))
                cleanPredicted = cleanStr(answer)

                predictedTokens = set([self.wnl.lemmatize(token) for token in cleanPredicted.split(" ")])
                correctTokens = set([self.wnl.lemmatize(token) for token in cleanCorrect.split(" ")])
                if correctTokens == {"no"}:
                    correctTokens.add("not")
                currentPrecision = len(predictedTokens.intersection(correctTokens)) / len(predictedTokens)
                currentRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens)
                currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
                f1.append(currentF1)
                recall.append(currentRecall)

                currentBleu = self.bleu([cleanPredicted], [[cleanCorrect]]).item()
                bleuScores.append(currentBleu)

                # If the question is closed, decide if it is correct or not
                if cleanCorrect in ["yes", "no"]:
                    closedQuestions.append(True)
                else:
                    closedQuestions.append(False)

                answersLog.append(
                    (
                        self.getCorrectAnswer(batch[idx]),
                        answer,
                        currentF1,
                        currentBleu,
                        currentRecall,
                        correctTokens,
                        predictedTokens,
                    )
                )

        # Compute the accuracy for closed questions
        closedQuestionsCorrect = 0
        openQuestionsRecall = []
        openQuestionsAccuracy = 0
        for idx, isClosed in enumerate(closedQuestions):
            if isClosed and recall[idx] >= 0.4:
                closedQuestionsCorrect += 1
            elif not isClosed:
                openQuestionsRecall.append(recall[idx])
                if recall[idx] >= 0.75:
                    openQuestionsAccuracy += 1

        metrics = {
            "bleu": sum(bleuScores) / len(bleuScores),
            "F1": sum(f1) / len(f1),
            "recall": sum(recall) / len(recall),
            "closedQuestionsAccuracy": closedQuestionsCorrect / sum(closedQuestions),
            "openQuestionsRecall": sum(openQuestionsRecall) / len(openQuestionsRecall),
            "openQuestionsAccuracy": openQuestionsAccuracy / len(openQuestionsRecall),
        }

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    @abstractmethod
    def getCorrectAnswer(self, sample):
        pass


class ImageClassification(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "Image Classification"
        self.scoringType = "multiclass"
        self.num_classes = None

    def run(self, params: EvalParams, batcher):
        self.logger.info(f"***** Benchmarking : {self.taskName} *****")

        predictions = []
        groundTruth = []
        answersLog = []

        scorerArgs = {"task": self.scoringType, "average": "macro"}
        scorerArgs.update(
            {"num_classes": self.num_classes} if self.scoringType == "multiclass" else {"num_labels": self.num_classes}
        )
        f1Scorer = F1Score(**scorerArgs)
        aurocScorer = AUROC(**scorerArgs)
        accuracy = Accuracy(**scorerArgs)

        dataloader = DataLoader(
            self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x
        )
        for batch in tqdm_logging(
            self.logger,
            dataloader,
            desc="Running inference",
        ):
            batchPrompts = []
            for sample in batch:
                text, img = self.format_question(sample)
                if params.fewshot and self.getPrompt() is not None:
                    batchPrompts.append((self.getPrompt()[0] + text, self.getPrompt()[1] + img))
                else:
                    batchPrompts.append((text, img))

            answers = batcher(batchPrompts)

            correctAnswers = [self.getCorrectAnswer(sample) for sample in batch]

            for idx, answer in enumerate(answers):
                pred = self.getPredictedAnswer(answer)
                gt = correctAnswers[idx]

                predictions.append(pred)
                groundTruth.append(gt)

                answersLog.append(
                    (
                        f"{self.getCorrectAnswer(batch[idx], fullText=True)} (index {gt})",
                        f"{answer} (index {pred})",
                        gt == pred,
                    )
                )

            # break

        # Convert pred and gt to tensor
        predictions = torch.tensor(predictions)
        if self.scoringType == "multiclass":
            predictions = torch.nn.functional.one_hot(predictions, num_classes=self.num_classes)
        predictions = predictions.to(torch.float32)
        groundTruth = torch.tensor(groundTruth)

        f1Macro = f1Scorer(predictions, groundTruth).item()
        auroc = aurocScorer(predictions, groundTruth).item()
        acc = accuracy(predictions, groundTruth).item()

        metrics = {"AUC-macro": auroc, "F1-macro": f1Macro, "Accuracy": acc}

        # Compute the scores
        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    @abstractmethod
    def getCorrectAnswer(self, sample, fullText=False) -> int:
        pass

    @abstractmethod
    def format_question(self, sample, prompt=False):
        pass

    @abstractmethod
    def getPredictedAnswer(self, answer) -> int:
        """Converts the free form text output to the answer index

        Args:
            sample (_type_): The sample used to generate the answer
            answer (_type_): The free form text output of the model

        Returns:
            int: The index of the answer
        """
        pass


class ReportComparison(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2, weights=[1 / 2, 1 / 2])
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys=("rougeL", "rouge1"))

    def run(self, params: EvalParams, batcher):
        self.logger.info(f"***** Benchmarking : {self.taskName} *****")
        refReports = []
        hypReports = []

        # Run the batcher for all data split in chunks
        dataloader = DataLoader(
            self.dataset, batch_size=params.batch_size, num_workers=params.num_workers, collate_fn=lambda x: x
        )
        for batch in tqdm_logging(self.logger, dataloader, desc="Generating reports"):
            batcherCorrect = [self.getCorrectAnswer(sample) for sample in batch]
            batcherHyp = batcher([self.format_question(sample) for sample in batch])
            batcherHyp = [h if h != "" else "Invalid Response" for h in batcherHyp]

            refReports += batcherCorrect
            hypReports += batcherHyp

        (
            bleu1Scores,
            bleu2Scores,
            bleu4Scores,
            rougeLScores,
            rouge1Scores,
            f1_bertscore_unscaled,
            chexbert_similarity,
            f1_radgraph,
            radcliq_v0_scores,
            meteor_scores,
            f1_bertscore,
        ) = self._evaluate_reports(hypReports, refReports)

        metrics = {
            "bleu1": sum(bleu1Scores) / len(bleu1Scores),
            "bleu4": sum(bleu4Scores) / len(bleu4Scores),
            "f1-radgraph": sum(f1_radgraph) / len(f1_radgraph),
            "CheXBert vector similarity": sum(chexbert_similarity) / len(chexbert_similarity),
            "f1-bertscore": sum(f1_bertscore_unscaled) / len(f1_bertscore_unscaled),
            "radcliq": sum(radcliq_v0_scores) / len(radcliq_v0_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
            "rougeL": sum(rougeLScores) / len(rougeLScores),
            "rouge1": sum(rouge1Scores) / len(rouge1Scores),
        }

        answersLog = zip(refReports, hypReports, bleu1Scores, bleu4Scores, rougeLScores)
        # Add a header to the log
        answersLog = [("ref", "hyp", "bleu1", "bleu4", "rougeL")] + list(answersLog)

        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def compute_chexbert(self, hypReports, refReports):
        df = pd.DataFrame(columns=["Report Impression"], data=refReports)
        labelsReference = self.engine.encoder(df)

        df = pd.DataFrame(columns=["Report Impression"], data=hypReports)
        labelsHypothesis = self.engine.encoder(df)

        # Compute the vector similarity between the reference and the geenrated reports
        return torch.cosine_similarity(labelsReference, labelsHypothesis)

    def compute_radgraph(self, hypReports, refReports):
        f1_radgraph = []
        for hyp, ref in zip(hypReports, refReports):
            # Compute the F1-radgraph score
            (_, _, hyp_annotation_lists, ref_annotation_lists) = self.engine.radgraph(refs=[ref], hyps=[hyp])
            f1_radgraph.append(
                exact_entity_token_if_rel_exists_reward(hyp_annotation_lists[0], ref_annotation_lists[0])
            )
        return torch.tensor(f1_radgraph)

    def _evaluate_reports(self, hypReports, refReports):
        for i in range(len(hypReports)):
            if hypReports[i] == "":
                hypReports[i] = "Invalid Response"
            if refReports[i] == "":
                refReports[i] = "Invalid Response"

        bleu1Scores = []
        bleu2Scores = []
        bleu4Scores = []
        rougeLScores = []
        rouge1Scores = []

        for hyp, ref in zip(hypReports, refReports):
            bleu1Scores.append(self.bleu_1([hyp], [[ref]]).item())
            bleu2Scores.append(self.bleu_2([hyp], [[ref]]).item())
            bleu4Scores.append(self.bleu_4([hyp], [[ref]]).item())
            currentRouge = self.rougeL([hyp], [[ref]])
            rougeLScores.append(currentRouge["rougeL_fmeasure"].item())
            rouge1Scores.append(currentRouge["rouge1_fmeasure"].item())

        f1_bertscore = compute_bertscore(hypReports, refReports)
        f1_bertscore_unscaled = compute_bertscore(hypReports, refReports, rescale=False)

        chexbert_similarity = self.compute_chexbert(hypReports, refReports)

        f1_radgraph = self.compute_radgraph(hypReports, refReports)

        bleu_scores = torch.tensor(bleu2Scores)

        radcliq_v0_scores = compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph)

        meteor_scores = compute_meteor(hypReports, refReports)

        return (
            bleu1Scores,
            bleu2Scores,
            bleu4Scores,
            rougeLScores,
            rouge1Scores,
            f1_bertscore_unscaled.tolist(),
            chexbert_similarity.tolist(),
            f1_radgraph.tolist(),
            radcliq_v0_scores.tolist(),
            meteor_scores,
            f1_bertscore.tolist(),
        )


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
    with open(os.path.join(module_path, "radcliq-v1_dill.pkl"), "rb") as f:
        composite_metric_v0_model: CompositeMetric = pickle.load(f)

    # The column need to be in the order [bleu, bertscore, chexbert, radgraph]
    input_data = torch.stack(
        [f1_radgraph, f1_bertscore, chexbert_similarity, bleu_scores],
        dim=1,
    )
    return composite_metric_v0_model.predict(input_data)
