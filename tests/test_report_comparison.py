"""Tests for the ReportComparison task."""

import csv
import json
import logging
import os
import zipfile

import numpy as np
from appdirs import user_cache_dir
from scipy.stats import kendalltau
from sklearn.utils import resample

from multimedeval import MultiMedEval, SetupParams
from multimedeval.utils import download_file

logging.basicConfig(level=logging.INFO)


def compute_kendall_tau(computed_scores, evaluator_scores):
    """Compute Kendall Tau."""
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
def test_report_comparison():
    """Tests the ReportComparison task."""
    engine = MultiMedEval()

    config_file_name = (
        "tests/test_config.json" if IN_GITHUB_ACTIONS else "MedMD_config.json"
    )
    with open(config_file_name, "r", encoding="utf-8") as file:
        config = json.load(file)
    if IN_GITHUB_ACTIONS:
        config["physionet_username"] = os.getenv("PHYSIONET_USERNAME")
        config["physionet_password"] = os.getenv("PHYSIONET_PASSWORD")

    try:
        success = engine.setup(
            SetupParams(
                chexbert_dir=config["chexbert_dir"],
                physionet_username=config["physionet_username"],
                physionet_password=config["physionet_password"],
            )
        )
    except Exception as e:
        raise AssertionError(f"Error in setup. {e}") from e

    print(f"Radgraph: {success['RadGraph']}")
    model_path = os.path.join(user_cache_dir("radgraph"))

    # Print the names of the files in the model_path folder and the size of each file
    print(f"Model path contains: {os.listdir(model_path)}")
    for file in os.listdir(model_path):
        print(f"{file}: {os.path.getsize(os.path.join(model_path, file))}")

    report_comparison = engine.name_to_task["MIMIC-CXR Report Generation"]

    # Directory to save the files
    directory = "tests"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    username, password = engine.get_physionet_credentials()

    download_file(
        "https://physionet.org/content/rexval-dataset/get-zip/1.0.0/",
        os.path.join("tests", "rexval.zip"),
        username,
        password,
    )

    # Unzip the file
    with zipfile.ZipFile(os.path.join("tests", "rexval.zip"), "r") as zip_ref:
        zip_ref.extractall("tests")

    # Load the all the report pairs from the csv file
    report_pairs = []
    id_to_details = {}
    current_id = 0
    with open(
        "tests/radiology-report-expert-evaluation-rexval-dataset-1.0.0/"
        "50_samples_gt_and_candidates.csv",
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            id_to_details[len(report_pairs)] = {"id": current_id, "pair": "radgraph"}
            report_pairs.append((row[1], row[2]))

            id_to_details[len(report_pairs)] = {"id": current_id, "pair": "bertscore"}
            report_pairs.append((row[1], row[3]))

            id_to_details[len(report_pairs)] = {"id": current_id, "pair": "s_emb"}
            report_pairs.append((row[1], row[4]))

            id_to_details[len(report_pairs)] = {"id": current_id, "pair": "bleu"}
            report_pairs.append((row[1], row[5]))

            current_id += 1

    id_to_num_error = {}
    with open(
        "tests/radiology-report-expert-evaluation-rexval-dataset-1.0.0/"
        "6_valid_raters_per_rater_error_categories.csv",
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            id_to_num_error[f"{row[0]}_{row[1]}"] = id_to_num_error.get(
                f"{row[0]}_{row[1]}", 0.0
            ) + float(row[5])

    for key in id_to_num_error:
        id_to_num_error[key] /= 6

    hyp_reports, ref_teports = zip(*report_pairs)
    print(f"Hypotheses: {hyp_reports[:5]}")
    print(f"References: {ref_teports[:5]}")
    (
        _,
        bleu2_scores,
        _,
        _,
        _,
        _,
        chexbert_similarity,
        f1_radgraph,
        radcliq_v0_scores,
        _,
        f1_bertscore,
    ) = report_comparison._evaluate_reports(hyp_reports, ref_teports)

    bleu_evaluator_and_computed_scores = []
    bertscore_evaluator_and_computed_scores = []
    chexbert_evaluator_and_computed_scores = []
    radgraph_evaluator_and_computed_scores = []
    radcliq_evaluator_and_computed_scores = []
    for i in range(len(report_pairs)):
        details = id_to_details[i]

        bleu_evaluator_and_computed_scores.append(
            (1 - bleu2_scores[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        bertscore_evaluator_and_computed_scores.append(
            (1 - f1_bertscore[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        chexbert_evaluator_and_computed_scores.append(
            (
                1 - chexbert_similarity[i],
                id_to_num_error[f"{details['id']}_{details['pair']}"],
            )
        )
        radgraph_evaluator_and_computed_scores.append(
            (1 - f1_radgraph[i], id_to_num_error[f"{details['id']}_{details['pair']}"])
        )
        radcliq_evaluator_and_computed_scores.append(
            (
                radcliq_v0_scores[i],
                id_to_num_error[f"{details['id']}_{details['pair']}"],
            )
        )

    # Compute kendall tau for BLEU
    computed_scores, evaluator_scores = zip(*bleu_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(
        f"Kendall Tau for BLEU: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]"
    )
    assert 0.368 < tau < 0.539

    # Compute kendall tau for BERTScore
    computed_scores, evaluator_scores = zip(*bertscore_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(
        f"Kendall Tau for BERTScore: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]"
    )
    assert 0.429 < tau < 0.584

    # Compute kendall tau for CheXBert
    computed_scores, evaluator_scores = zip(*chexbert_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(
        f"Kendall Tau for CheXBert: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]"
    )
    assert 0.417 < tau < 0.576

    # Compute kendall tau for RadGraph
    computed_scores, evaluator_scores = zip(*radgraph_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(
        f"Kendall Tau for RadGraph: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]"
    )
    assert 0.449 < tau < 0.578

    # Compute kendall tau for RadCliq
    computed_scores, evaluator_scores = zip(*radcliq_evaluator_and_computed_scores)
    tau, tau_low, tau_high = compute_kendall_tau(computed_scores, evaluator_scores)
    print(
        f"Kendall Tau for RadCliq: {tau}, 95% confidence interval: [{tau_low}, {tau_high}]"
    )
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
    test_report_comparison()
