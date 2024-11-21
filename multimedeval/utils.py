"""Utility functions for the MultimedEval library."""

import csv
import json
import logging
import os
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Literal

import numpy as np
import requests
import torch
from datasets import Dataset
from tqdm import tqdm
from PIL.Image import Image

if TYPE_CHECKING:
    from multimedeval import MultiMedEval


class Benchmark(ABC):
    """Abstract class for benchmarks."""

    def __init__(self, engine, logger) -> None:
        """Initialize the benchmark.

        Args:
            engine: Reference to the engine class.
            logger: A logger object.
        """
        self.task_name: str = "None"
        self.engine: MultiMedEval = engine
        self.modality: str = "None"
        self.task: str = "None"
        self._prompt = None
        self.train_dataset = None
        self.dataset: Optional[Dataset] = None
        self.logger: logging.Logger = logger

    def get_prompt(self):
        """Get the fewshot prompt."""
        if not self.train_dataset:
            return None

        if self._prompt is None:
            batcher_inputs = []
            for i in range(5):
                index = int(i / 5 * len(self.train_dataset))
                input = self.format_question(
                    self.train_dataset[index],
                    prompt=True,
                )
                batcher_inputs.append(input)
            self._prompt = batcher_inputs
        return self._prompt

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)

    @abstractmethod
    def format_question(self, sample, prompt=False):
        """Format the question in a Huggingface format."""

    @abstractmethod
    def setup(self):
        """Setup the benchmark and download the dataset."""

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx: The index of the item to get.

        Returns:
            The item from the dataset.
        """
        return {"idx": idx, "sample": self.dataset[idx]}

    @abstractmethod
    def evaluate(self, predictions):
        """Runs the evaluation on the predictions."""


@dataclass
class EvalParams:
    """Dataclass defining the parameters for evaluation.

    Args:
        batch_size: The size of the batches sent to the user's batcher Callable.
        run_name: The name to use for the folder where the output will be stored.
        fewshot: A boolean indicating whether the evaluation is few-shot.
        num_workers: The number of workers for the dataloader.
        device: The device to run the evaluation on.
        tensorBoardWriter: The tensorboard writer to use for logging.
        tensorboardStep: The global step for logging to tensorboard.

    Raises:
        ImportError: raises an import error if tensorboard is not installed.
    """

    batch_size: int = 128
    run_name: str = f"run {datetime.now()}"
    fewshot: bool = False
    num_workers: int = 0
    tensorboard_writer: Any = None
    tensorboard_step: int = 0
    mimic_cxr_include_indication_section: bool = False
    dataloader_fn: Any = None

    def __post_init__(self):
        """Check if tensorboard is installed."""
        if self.tensorboard_writer is not None:
            try:
                from torch.utils.tensorboard import (  # noqa # pylint: disable=unused-import, import-outside-toplevel
                    SummaryWriter,
                )
            except ImportError as e:
                raise ImportError(
                    f"Please install tensorboard using `pip install tensorboard` {e}"
                ) from e


@dataclass
class SetupParams:
    """Parameter dataclass for setting up the benchmark.

    Args:
        medqa_dir: The path to the MedQA dataset.
        pubmedqa_dir: The path to the PubMedQA dataset.
        medmcqa_dir: The path to the MedMCQA dataset.
        vqa_rad_dir: The path to the VQA-RAD dataset.
        path_vqa_dir: The path to the Path-VQA dataset.
        slake_dir: The path to the SLAKE dataset.
        mimic_iii_dir: The path to the MIMIC-III dataset.
        mednli_dir: The path to the MedNLI dataset.
        mimic_cxr_dir: The path to the MIMIC-CXR dataset.
        vindr_mammo_dir: The path to the VinDr-Mammo dataset.
        pad_ufes_20_dir: The path to the PadChest dataset.
        cbis_ddsm_dir: The path to the CBIS-DDSM dataset.
        mnist_oct_dir: The path to the MNIST-OCT dataset.
        mnist_path_dir: The path to the MNIST-Path dataset.
        mnist_blood_dir: The path to the MNIST-Blood dataset.
        mnist_breast_dir: The path to the MNIST-Breast dataset.
        mnist_derma_dir: The path to the MNIST-Derma dataset.
        mnist_organc_dir: The path to the MNIST-OrganC dataset.
        mnist_organs_dir: The path to the MNIST-OrganS dataset.
        mnist_pneumonia_dir: The path to the MNIST-Pneumonia dataset.
        mnist_retina_dir: The path to the MNIST-Retina dataset.
        mnist_tissue_dir: The path to the MNIST-Tissue dataset.
        chestxray14_dir: The path to the ChestXray14 dataset.
        chexbert_dir: The path to the CheXpert dataset.
        physionet_username: The username for the physionet dataset.
        physionet_password: The password for the physionet dataset.

    """

    medqa_dir: Optional[Union[str, os.PathLike]] = None
    pubmedqa_dir: Optional[Union[str, os.PathLike]] = None
    medmcqa_dir: Optional[Union[str, os.PathLike]] = None
    vqa_rad_dir: Optional[Union[str, os.PathLike]] = None
    path_vqa_dir: Optional[Union[str, os.PathLike]] = None
    slake_dir: Optional[Union[str, os.PathLike]] = None
    mimic_iii_dir: Optional[Union[str, os.PathLike]] = None
    mednli_dir: Optional[Union[str, os.PathLike]] = None
    mimic_cxr_dir: Optional[Union[str, os.PathLike]] = None
    vindr_mammo_dir: Optional[Union[str, os.PathLike]] = None
    pad_ufes_20_dir: Optional[Union[str, os.PathLike]] = None
    cbis_ddsm_dir: Optional[Union[str, os.PathLike]] = None
    mnist_oct_dir: Optional[Union[str, os.PathLike]] = None
    mnist_path_dir: Optional[Union[str, os.PathLike]] = None
    mnist_blood_dir: Optional[Union[str, os.PathLike]] = None
    mnist_breast_dir: Optional[Union[str, os.PathLike]] = None
    mnist_derma_dir: Optional[Union[str, os.PathLike]] = None
    mnist_organc_dir: Optional[Union[str, os.PathLike]] = None
    mnist_organs_dir: Optional[Union[str, os.PathLike]] = None
    mnist_pneumonia_dir: Optional[Union[str, os.PathLike]] = None
    mnist_retina_dir: Optional[Union[str, os.PathLike]] = None
    mnist_tissue_dir: Optional[Union[str, os.PathLike]] = None
    diff_vqa_dir: Optional[Union[str, os.PathLike]] = None
    mmlu_dir: Optional[Union[str, os.PathLike]] = None
    chestxray14_dir: Optional[Union[str, os.PathLike]] = None
    chexbert_dir: Optional[Union[str, os.PathLike]] = None
    physionet_username: Optional[str] = None
    physionet_password: Optional[str] = None
    device: Optional[str] = "cuda"

    def __post_init__(self):
        """Checking that the device is available."""
        if self.device != "cpu":
            # Check if the device is available (handle cuda and mps)
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"

            elif self.device == "mps" and not torch.backends.mps.is_available():
                self.device = "cpu"


@dataclass
class EvaluationOutput:
    """Dataclass for storing the output of the evaluation."""

    metrics: Dict[str, float]
    answer_log: Optional[List[tuple]] = None

@dataclass
class BatcherInput:
    """Dataclass for unified formatting of the basic (conversation, images, seg(optional)) batcher input"""
    
    conversation: List[dict] = field(default_factory = list)
    images: Optional[List[Image]] = field(default_factory = list)
    segmentation_masks: Optional[List[Image]] = field(default_factory = list)

    def _add_text_prompt(self, role: Literal["assistant", "user", "system"], content: str):
        self.conversation.append({
            "role": role,
            "content": content
        })

    def _add_images(self, image: Image):
        self.images.append(image)
    
    def _add_segmentation_mask(self, seg_mask: Image):
        self.segmentation_masks.append(seg_mask)

def remove_punctuation(input_string: str):
    """Removes punctuation from a string.

    Args:
        input_string: The string to remove punctuation from.

    Returns:
        The string with punctuation removed.
    """
    # Make a translator object to replace punctuation with none
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    # Use the translator
    return input_string.translate(translator)


def csv_writer(data, path):
    """Writes data to a csv file.

    Args:
        data: The data to write (list of lists).
        path: The path to write the data to.
    """
    try:
        with open(f"{path}.csv", "w", newline="", encoding="utf-8") as f:
            spam_writer = csv.writer(f)
            spam_writer.writerows(data)
    except IOError as e:
        print(e)


def json_writer(data, path):
    """Writes data to a json file.

    Args:
        data: The serializable data to write.
        path: The path to write the data to.
    """
    try:
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
    except IOError as e:
        print(e)


SUPPORTED_FILETYPES = {"csv": csv_writer, "json": json_writer}


def file_writer_factory(file_type):
    """Factory function for file writers.

    Args:
        fileType: The type of file to write.

    Returns:
        The file writer function.
    """
    assert file_type in SUPPORTED_FILETYPES, f"{file_type} not supported."

    return SUPPORTED_FILETYPES[file_type]


def _exact_entity_token_if_rel_exists_reward(
    hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            1
            for x in hypothesis_relation_token_list
            if (x in reference_relation_token_list)
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            1
            for x in reference_relation_token_list
            if (x in hypothesis_relation_token_list)
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


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


def section_text(text):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section,
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(r"\n ([A-Z ()/,-]+):\s", re.DOTALL)

    sections = []
    section_names = []
    section_idx = []

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0 : s.start(1)])
        section_names.append("preamble")
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find("\n")
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append("full report")
        section_idx.append(0)

    section_names = _normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ("impression", "findings"):
            if sections[i].strip() == "":
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ("impression" not in section_names) & ("findings" not in section_names):
        # create a new section for the final paragraph
        if "\n \n" in sections[-1]:
            sections.append("\n \n".join(sections[-1].split("\n \n")[1:]))
            sections[-2] = sections[-2].split("\n \n")[0]
            section_names.append("last_paragraph")
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def _normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    frequent_sections = {
        "preamble": "preamble",  # 227885
        "impression": "impression",  # 187759
        "comparison": "comparison",  # 154647
        "indication": "indication",  # 153730
        "findings": "findings",  # 149842
        "examination": "examination",  # 94094
        "technique": "technique",  # 81402
        "history": "history",  # 45624
        "comparisons": "comparison",  # 8686
        "clinical history": "history",  # 7121
        "reason for examination": "indication",  # 5845
        "notification": "notification",  # 5749
        "reason for exam": "indication",  # 4430
        "clinical information": "history",  # 4024
        "exam": "examination",  # 3907
        "clinical indication": "indication",  # 1945
        "conclusion": "impression",  # 1802
        "chest, two views": "findings",  # 1735
        "recommendation(s)": "recommendations",  # 1700
        "type of examination": "examination",  # 1678
        "reference exam": "comparison",  # 347
        "patient history": "history",  # 251
        "addendum": "addendum",  # 183
        "comparison exam": "comparison",  # 163
        "date": "date",  # 108
        "comment": "comment",  # 88
        "findings and impression": "impression",  # 87
        "wet read": "wet read",  # 83
        "comparison film": "comparison",  # 79
        "recommendations": "recommendations",  # 72
        "findings/impression": "impression",  # 47
        "pfi": "history",
        "recommendation": "recommendations",
        "wetread": "wet read",
        "ndication": "impression",  # 1
        "impresson": "impression",  # 2
        "imprression": "impression",  # 1
        "imoression": "impression",  # 1
        "impressoin": "impression",  # 1
        "imprssion": "impression",  # 1
        "impresion": "impression",  # 1
        "imperssion": "impression",  # 1
        "mpression": "impression",  # 1
        "impession": "impression",  # 3
        "findings/ impression": "impression",  # ,1
        "finding": "findings",  # ,8
        "findins": "findings",
        "findindgs": "findings",  # ,1
        "findgings": "findings",  # ,1
        "findngs": "findings",  # ,1
        "findnings": "findings",  # ,1
        "finidngs": "findings",  # ,2
        "idication": "indication",  # ,1
        "reference findings": "findings",  # ,1
        "comparision": "comparison",  # ,2
        "comparsion": "comparison",  # ,1
        "comparrison": "comparison",  # ,1
        "comparisions": "comparison",  # ,1
    }

    p_findings = [
        "chest",
        "portable",
        "pa and lateral",
        "lateral and pa",
        "ap and lateral",
        "lateral and ap",
        "frontal and",
        "two views",
        "frontal view",
        "pa view",
        "ap view",
        "one view",
        "lateral view",
        "bone window",
        "frontal upright",
        "frontal semi-upright",
        "ribs",
        "pa and lat",
    ]
    p_findings = re.compile(f"({'|'.join(p_findings)})")

    main_sections = ["impression", "findings", "history", "comparison", "addendum"]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = "findings"

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


def download_file(url: str, fname: str, username=None, password=None):
    """Download a file from a URL.

    Args:
        url: The URL to download the file from.
        fname: The name of the file to save the download to.
        username: Physionet username. Defaults to None.
        password: Physionet password. Defaults to None.
    """
    header = {
        "User-Agent": "Wget/1.20.3 (linux-gnu)",
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "Host": "physionet.org",
        "Connection": "Keep-Alive",
        "Proxy-Connection": "Keep-Alive",
        "Cookie": "testcookie=1",
    }
    auth = (username, password)
    chunk_size = 1024

    resp = requests.get(url, stream=True, auth=auth, headers=header)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as progress_bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            progress_bar.update(size)


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")  # noqa
comma_strip = re.compile(r"(\d)(\,)(\d)")  # noqa
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def clean_str(token):
    """Cleans a string (removes punctuation, lowers...).

    Args:
        token: The string to clean.

    Returns:
        The cleaned string.
    """
    token = token.lower()
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) is not None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token
