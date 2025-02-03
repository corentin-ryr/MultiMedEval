import json
import re
import traceback
import os
from huggingface_hub import hf_hub_download, list_repo_files
from nltk.tokenize import wordpunct_tokenize
import torch


def batch_to_device(inp, device):
    if isinstance(inp, torch.Tensor):
        return inp.to(device)
    elif isinstance(inp, dict):
        return {k: batch_to_device(v, device) for k, v in inp.items()}
    elif isinstance(inp, list):
        return [batch_to_device(v, device) for v in inp]
    else:
        return inp


def download_model(repo_id, cache_dir, filename):
    # creating cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Download
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    except Exception as e:
        raise Exception("Cannot download the file {}".format(e))
    return path


def radgraph_xl_preprocess_report(text):
    text = text.replace("\\n", "  ")
    text = text.replace("\\f", "  ")
    text = text.replace("\\u2122", "      ")
    text = text.replace("\n", " ")
    text = text.replace("\\\"", "``")

    text_sub = re.sub(r'\s+', ' ', text)
    tokenized_text = " ".join(wordpunct_tokenize(text_sub))
    tokenized_text = tokenized_text.replace(").", ") .")
    tokenized_text = tokenized_text.replace("%.", "% .")
    tokenized_text = tokenized_text.replace(".'", ". '")
    tokenized_text = tokenized_text.replace("%,", "% ,")
    tokenized_text = tokenized_text.replace("%)", "% )")
    return tokenized_text


def echograph_preprocess_report(text):
    # Split the text into words
    words = text.split()
    # Remove the period at the end of each word, if present
    words = [word.rstrip('.') for word in words]
    # Initialize a list to store the tokenized words
    tokenized_words = []
    # Loop through each word in the list of words
    for word in words:
        # Check if the word contains the pattern "/"
        if "/" in word:
            # Split the word based on the "/" symbol
            parts = word.split("/")
            # Add the split parts to the tokenized words list
            tokenized_words.extend(parts)
        # Check if the word contains the pattern ">"
        elif ">" in word:
            # Split the word based on the ">" symbol
            parts = word.split(">")
            # Add the first part to the tokenized words list
            tokenized_words.append(parts[0])
            # Add the second part to the tokenized words list with the ">" symbol
            tokenized_words.append(">" + parts[1])
        else:
            # If the word does not contain "/" or ">", add it directly to the tokenized words list
            tokenized_words.append(word)
    return tokenized_words


def preprocess_reports(report_list, model_type):
    """Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    reports = []
    for idx, report in enumerate(report_list):
        if model_type == "radgraph":
            sen = re.sub(
                "(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )", r" ", report
            ).split()
        elif model_type == "radgraph-xl":
            sen = radgraph_xl_preprocess_report(report).split()
        elif model_type == "echograph":
            sen = echograph_preprocess_report(report)
        else:
            raise NotImplementedError(model_type)

        temp_dict = {"doc_key": str(idx), "sentences": [sen], "dataset": model_type}
        reports.append(temp_dict)

    return reports


def postprocess_reports(results, data_source=None):
    """Post processes all the reports and saves the result in JSON format

    Args:
        results: List of output dicts for individual reports
        data_source: Source of data, if any

    Returns:
        A JSON string representing the final processed reports
    """
    final_dict = {}

    for file in results:
        try:
            temp_dict = {}

            temp_dict["text"] = " ".join(file["sentences"][0])
            n = file["predicted_ner"][0]
            r = file["predicted_relations"][0]
            s = file["sentences"][0]
            temp_dict["entities"] = get_entity(n, r, s)
            temp_dict["data_source"] = data_source
            temp_dict["data_split"] = "inference"

            final_dict[file["doc_key"]] = temp_dict

        except Exception:
            traceback.print_exc()
            print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")

    return final_dict


def get_entity(n, r, s):
    """Gets the entities for individual reports

    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence

    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json

    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict["tokens"] = " ".join(s[start_idx: end_idx + 1])
        temp_dict["label"] = label
        temp_dict["start_ix"] = start_idx
        temp_dict["end_ix"] = end_idx
        rel = []
        relation_idx = [
            i for i, val in enumerate(rel_list) if val == [start_idx, end_idx]
        ]
        for i, val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab, str(object_idx)])
        temp_dict["relations"] = rel
        dict_entity[str(idx + 1)] = temp_dict

    return dict_entity
