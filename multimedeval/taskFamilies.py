
from multimedeval.utils import Benchmark, EvalParams, cleanStr
from torchmetrics.text import BLEUScore
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod
from nltk.stem import WordNetLemmatizer
from torchmetrics import F1Score, AUROC, Accuracy
import torch



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
        for batch in tqdm(
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
        for batch in tqdm(
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
        for batch in tqdm(
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