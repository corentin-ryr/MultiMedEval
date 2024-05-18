from multimedeval.utils import Benchmark, cleanStr, exact_entity_token_if_rel_exists_reward, EvaluationOutput
from torchmetrics.text import BLEUScore, ROUGEScore
from abc import abstractmethod
from nltk.stem import WordNetLemmatizer
from torchmetrics import F1Score, AUROC, Accuracy
import torch
import pandas as pd
from multimedeval.reportComparisonUtils import compute_bertscore, compute_composite, compute_meteor


class QA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = "QA"

    @abstractmethod
    def getCorrectAnswer(self, sample, fullText=False):
        pass

    @abstractmethod
    def getPredictedAnswer(self, pred: str, sample):
        pass

    def evaluate(self, predictions):
        correct_answers = 0
        total_answers = 0

        answersLog = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self.__getitem__(idx)["sample"]

            gold = self.getCorrectAnswer(sample)
            pred = self.getPredictedAnswer(answer, sample)
            if pred == gold:
                correct_answers += 1
            total_answers += 1

            answersLog.append((self.getCorrectAnswer(sample, fullText=True), answer, pred, gold, pred == gold))

            # break

        # TODO: add others metrics such as AUC, F1...
        metrics = {"accuracy": correct_answers / total_answers}

        return EvaluationOutput(answer_log=answersLog, metrics=metrics)


class VQA(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "VQA"
        self.wnl = WordNetLemmatizer()

    def _preprocess(self, text):
        tokenizedText = set(text.split())
        tokenizedText.discard("")
        return tokenizedText

    @abstractmethod
    def getCorrectAnswer(self, sample):
        pass

    def evaluate(self, predictions):
        answersLog = [["correct", "predicted", "F1", "BLEU", "recall", "correct tokens", "predicted tokens"]]
        bleuScores = []
        f1 = []
        recall = []
        closedQuestions = []
        closedQuestionsCorrect = 0

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self.__getitem__(idx)["sample"]

            # Compute the number of tokens recalled in the answer
            cleanCorrect = cleanStr(self.getCorrectAnswer(sample))
            cleanPredicted = cleanStr(answer)

            predictedTokens = self._preprocess(cleanPredicted)
            correctTokens = self._preprocess(cleanCorrect)

            currentPrecision = (
                len(predictedTokens.intersection(correctTokens)) / len(predictedTokens)
                if len(predictedTokens) > 0
                else 0
            )
            currentRecall = (
                len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
            )
            currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
            f1.append(currentF1)
            recall.append(currentRecall)

            currentBleu = self.bleu([cleanPredicted], [[cleanCorrect]]).item()
            bleuScores.append(currentBleu)

            # If the question is closed, decide if it is correct or not
            if cleanCorrect in ["yes", "no"]:
                closedQuestions.append(True)

                if correctTokens == {"no"}:
                    correctTokens.add("not")

                closedQRecall = (
                    len(predictedTokens.intersection(correctTokens)) / len(correctTokens)
                    if len(correctTokens) > 0
                    else 0
                )
                if closedQRecall >= 0.4:
                    closedQuestionsCorrect += 1
            else:
                closedQuestions.append(False)

            answersLog.append(
                (
                    self.getCorrectAnswer(sample),
                    answer,
                    currentF1,
                    currentBleu,
                    currentRecall,
                    correctTokens,
                    predictedTokens,
                )
            )

        # Compute the accuracy for closed questions
        openQuestionsRecall = []
        openQuestionsAccuracy = 0
        for idx, isClosed in enumerate(closedQuestions):
            if not isClosed:
                openQuestionsRecall.append(recall[idx])
                if recall[idx] >= 0.75:
                    openQuestionsAccuracy += 1

        metrics = {
            "bleu": sum(bleuScores) / len(bleuScores) if len(bleuScores) > 0 else 0,
            "F1": sum(f1) / len(f1) if len(f1) > 0 else 0,
            "recall": sum(recall) / len(recall) if len(recall) > 0 else 0,
            "closedQuestionsAccuracy": closedQuestionsCorrect / sum(closedQuestions) if sum(closedQuestions) > 0 else 0,
            "openQuestionsRecall": (
                sum(openQuestionsRecall) / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
            "openQuestionsAccuracy": (
                openQuestionsAccuracy / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
        }

        return EvaluationOutput(answer_log=answersLog, metrics=metrics)


class ImageClassification(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bleu = BLEUScore(n_gram=1)
        self.task = "Image Classification"
        self.scoringType = "multiclass"
        self.num_classes = None

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

    def evaluate(self, predictions):

        scorerArgs = {"task": self.scoringType, "average": "macro"}
        scorerArgs.update(
            {"num_classes": self.num_classes} if self.scoringType == "multiclass" else {"num_labels": self.num_classes}
        )
        f1Scorer = F1Score(**scorerArgs)
        aurocScorer = AUROC(**scorerArgs)
        accuracy = Accuracy(**scorerArgs)

        predictedAnswers = []
        groundTruth = []
        answersLog = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self.__getitem__(idx)["sample"]

            pred = self.getPredictedAnswer(answer)
            gt = self.getCorrectAnswer(sample)

            predictedAnswers.append(pred)
            groundTruth.append(gt)

            answersLog.append(
                (f"{self.getCorrectAnswer(sample, fullText=True)} (index {gt})", f"{answer} (index {pred})", gt == pred)
            )

        # Convert pred and gt to tensor
        predictedAnswers = torch.tensor(predictedAnswers)
        if self.scoringType == "multiclass":
            predictedAnswers = torch.nn.functional.one_hot(predictedAnswers, num_classes=self.num_classes)
        predictedAnswers = predictedAnswers.to(torch.float32)
        groundTruth = torch.tensor(groundTruth)

        f1Macro = f1Scorer(predictedAnswers, groundTruth).item()
        auroc = aurocScorer(predictedAnswers, groundTruth).item()
        acc = accuracy(predictedAnswers, groundTruth).item()

        metrics = {"AUC-macro": auroc, "F1-macro": f1Macro, "Accuracy": acc}

        return EvaluationOutput(answer_log=answersLog, metrics=metrics)


class ReportComparison(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2, weights=[1 / 2, 1 / 2])
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys=("rougeL", "rouge1"))

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

    def evaluate(self, predictions):
        refReports = []
        hypReports = []

        for prediction in predictions:
            answer = prediction["answer"]
            idx = prediction["idx"]
            sample = self.__getitem__(idx)["sample"]

            refReports.append(self.getCorrectAnswer(sample))
            hypReports.append(answer)

        (
            bleu1Scores,
            _,
            bleu4Scores,
            rougeLScores,
            rouge1Scores,
            f1_bertscore_unscaled,
            chexbert_similarity,
            f1_radgraph,
            radcliq_v0_scores,
            meteor_scores,
            _,
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

        return EvaluationOutput(answer_log=answersLog, metrics=metrics)
