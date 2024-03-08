from multimedeval.mimic import MIMIC_CXR_reportgen
from multimedeval import MultiMedEval, SetupParams, EvalParams
import csv
import logging
from scipy.stats import kendalltau
import numpy as np
from sklearn.utils import resample
import os
import pytest
import json

logging.basicConfig(level=logging.INFO)


def compute_kendall_tau(computed_scores, evaluator_scores):
    tau, _ = kendalltau(evaluator_scores, computed_scores)

    num_samples = len(computed_scores)

    bootstrap_samples = np.zeros(num_samples)
    for i in range(num_samples):
        x_resampled, y_resampled = resample(computed_scores, evaluator_scores)
        tau_resampled, _ = kendalltau(x_resampled, y_resampled)
        bootstrap_samples[i] = tau_resampled

    # Calculate confidence interval
    confidence_interval = np.percentile(bootstrap_samples, [2.5, 97.5])
    return tau, confidence_interval[0], confidence_interval[1]

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_ReportComparison():
    engine = MultiMedEval()

    config = json.load(open("tests/test_config.json")) if IN_GITHUB_ACTIONS else json.load(open("MedMD_config.json"))

    engine.setup(SetupParams(CheXBert_dir=config["CheXBert_dir"]))
    reportComparison = engine.nameToTask["MIMIC-CXR Report Generation"]

    # Load the all the report pairs from the csv file
    reportPairs = []
    id_to_details = {}
    id = 0
    with open("tests/50_samples_gt_and_candidates.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            id_to_details[len(reportPairs)] = {"id": id, "pair": "radgraph"}
            reportPairs.append((row[1], row[2]))

            id_to_details[len(reportPairs)] = {"id": id, "pair": "bertscore"}
            reportPairs.append((row[1], row[3]))

            id_to_details[len(reportPairs)] = {"id": id, "pair": "s_emb"}
            reportPairs.append((row[1], row[4]))

            id_to_details[len(reportPairs)] = {"id": id, "pair": "bleu"}
            reportPairs.append((row[1], row[5]))

            id += 1

    id_to_num_error = {}
    with open("tests/6_valid_raters_per_rater_error_categories.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            id_to_num_error[f"{row[0]}_{row[1]}"] = id_to_num_error.get(f"{row[0]}_{row[1]}", 0.0) + float(row[5])

    for key in id_to_num_error:
        id_to_num_error[key] /= 6

    hypReports, refReports = zip(*reportPairs)
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
    ) = reportComparison._evaluate_reports(hypReports, refReports)

    BLEU_evaluator_and_computed_scores = []
    BERTScore_evaluator_and_computed_scores = []
    chexbert_evaluator_and_computed_scores = []
    radgraph_evaluator_and_computed_scores = []
    radcliq_evaluator_and_computed_scores = []
    for i in range(len(reportPairs)):
        details = id_to_details[i]

        BLEU_evaluator_and_computed_scores.append(
            (1 - bleu2Scores[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        BERTScore_evaluator_and_computed_scores.append(
            (1 - f1_bertscore[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        chexbert_evaluator_and_computed_scores.append(
            (1 - chexbert_similarity[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        radgraph_evaluator_and_computed_scores.append(
            (1 - f1_radgraph[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        radcliq_evaluator_and_computed_scores.append(
            (radcliq_v0_scores[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )

    # Compute kendall tau for BLEU
    computed_scores, evaluator_scores = zip(*BLEU_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(f"Kendall Tau for BLEU: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]")
    assert 0.368 < tau < 0.539

    # Compute kendall tau for BERTScore
    computed_scores, evaluator_scores = zip(*BERTScore_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(f"Kendall Tau for BERTScore: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]")
    assert 0.429 < tau < 0.584

    # Compute kendall tau for CheXBert
    computed_scores, evaluator_scores = zip(*chexbert_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(f"Kendall Tau for CheXBert: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]")
    assert 0.417 < tau < 0.576

    # Compute kendall tau for RadGraph
    computed_scores, evaluator_scores = zip(*radgraph_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(f"Kendall Tau for RadGraph: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]")
    assert 0.449 < tau < 0.578

    # Compute kendall tau for RadCliq
    computed_scores, evaluator_scores = zip(*radcliq_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(f"Kendall Tau for RadCliq: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]")
    assert 0.450 < tau < 0.749

    # Make a plot with 4 subplots

    # fig, axs = plt.subplots(1, 4, figsize=(40, 6))

    # # BLEU
    # computed_scores, evaluator_scores = zip(*BLEU_evaluator_and_computed_scores)
    # axs[0].scatter(evaluator_scores, computed_scores)
    # axs[0].set_xlabel("Total errors")
    # axs[0].set_ylabel("1 - score")
    # axs[0].set_title("BLEU")

    # # BERTScore
    # computed_scores, evaluator_scores = zip(*BERTScore_evaluator_and_computed_scores)
    # axs[1].scatter(evaluator_scores, computed_scores)
    # axs[1].set_xlabel("Total errors")
    # axs[1].set_ylabel("1 - score")
    # axs[1].set_title("BERTScore")

    # # CheXBert
    # computed_scores, evaluator_scores = zip(*chexbert_evaluator_and_computed_scores)
    # axs[2].scatter(evaluator_scores, computed_scores)
    # axs[2].set_xlabel("Total errors")
    # axs[2].set_ylabel("1 - score")
    # axs[2].set_title("CheXBert")

    # # RadGraph
    # computed_scores, evaluator_scores = zip(*radgraph_evaluator_and_computed_scores)
    # axs[3].scatter(evaluator_scores, computed_scores)
    # axs[3].set_xlabel("Total errors")
    # axs[3].set_ylabel("1 - score")
    # axs[3].set_title("RadGraph")

    # plt.savefig("scatter_plot.png")


if __name__ == "__main__":
    test_ReportComparison()
