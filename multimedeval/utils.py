from dataclasses import dataclass
import string
from datetime import datetime
import csv
import json
import numpy as np
import re
from abc import abstractmethod, ABC
import torch
import requests
from tqdm import tqdm
import re
import requests
import os
from typing import Optional
from typing import Any
import logging

class Benchmark(ABC):
    def __init__(self, engine, logger) -> None:
        self.taskName:str = "None"
        self.engine = engine
        self.modality:str = "None"
        self.task:str = "None"
        self._prompt = None
        self.trainDataset = None
        self.dataset = None
        self.logger:logging.Logger = logger

    def getPrompt(self):
        if not self.trainDataset:
            return None

        if self._prompt is None:
            prompt = []
            images = []
            for i in range(3):
                index = int(i / 3 * len(self.trainDataset))
                text, img = self.format_question(
                    self.trainDataset[index],
                    prompt=True,
                )
                prompt += text
                images += img
            self._prompt = (prompt, images)

        return self._prompt

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def run(self, params, batcher):
        pass

    @abstractmethod
    def format_question(self, sample, prompt=False):
        pass

    @abstractmethod
    def setup(self):
        pass


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
    batch_size: Optional[int] = 128
    run_name: Optional[str] = f"run {datetime.now()}"
    fewshot: Optional[bool] = False
    num_workers: Optional[int] = 0
    tensorboardWriter:Optional[Any] = None
    tensorboardStep: Optional[int] = 0
    mimic_cxr_include_indication_section: Optional[bool] = False

    def __post_init__(self):
        if self.tensorboardWriter is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError("Please install tensorboard using `pip install tensorboard`")

@dataclass
class SetupParams:
    """Parameter dataclass for setting up the benchmark.

    Args:
        MedQA_dir: The path to the MedQA dataset.
        PubMedQA_dir: The path to the PubMedQA dataset.
        MedMCQA_dir: The path to the MedMCQA dataset.
        VQA_RAD_dir: The path to the VQA-RAD dataset.
        Path_VQA_dir: The path to the Path-VQA dataset.
        SLAKE_dir: The path to the SLAKE dataset.
        MIMIC_III_dir: The path to the MIMIC-III dataset.
        MedNLI_dir: The path to the MedNLI dataset.
        MIMIC_CXR_dir: The path to the MIMIC-CXR dataset.
        VinDr_Mammo_dir: The path to the VinDr-Mammo dataset.
        Pad_UFES_20_dir: The path to the PadChest dataset.
        CBIS_DDSM_dir: The path to the CBIS-DDSM dataset.
        MNIST_Oct_dir: The path to the MNIST-OCT dataset.
        MNIST_Path_dir: The path to the MNIST-Path dataset.
        MNIST_Blood_dir: The path to the MNIST-Blood dataset.
        MNIST_Breast_dir: The path to the MNIST-Breast dataset.
        MNIST_Derma_dir: The path to the MNIST-Derma dataset.
        MNIST_OrganC_dir: The path to the MNIST-OrganC dataset.
        MNIST_OrganS_dir: The path to the MNIST-OrganS dataset.
        MNIST_Pneumonia_dir: The path to the MNIST-Pneumonia dataset.
        MNIST_Retina_dir: The path to the MNIST-Retina dataset.
        MNIST_Tissue_dir: The path to the MNIST-Tissue dataset.
        CheXBert_dir: The path to the CheXpert dataset.
        physionet_username: The username for the physionet dataset.
        physionet_password: The password for the physionet dataset.
        
    """


    MedQA_dir: Optional[str|os.PathLike] = None
    PubMedQA_dir: Optional[str|os.PathLike] = None
    MedMCQA_dir: Optional[str|os.PathLike] = None
    VQA_RAD_dir: Optional[str|os.PathLike] = None
    Path_VQA_dir: Optional[str|os.PathLike] = None
    SLAKE_dir: Optional[str|os.PathLike] = None
    MIMIC_III_dir: Optional[str|os.PathLike] = None
    MedNLI_dir: Optional[str|os.PathLike] = None
    MIMIC_CXR_dir: Optional[str|os.PathLike] = None
    VinDr_Mammo_dir: Optional[str|os.PathLike] = None
    Pad_UFES_20_dir: Optional[str|os.PathLike] = None
    CBIS_DDSM_dir: Optional[str|os.PathLike] = None
    MNIST_Oct_dir: Optional[str|os.PathLike] = None
    MNIST_Path_dir: Optional[str|os.PathLike] = None
    MNIST_Blood_dir: Optional[str|os.PathLike] = None
    MNIST_Breast_dir: Optional[str|os.PathLike] = None
    MNIST_Derma_dir: Optional[str|os.PathLike] = None
    MNIST_OrganC_dir: Optional[str|os.PathLike] = None
    MNIST_OrganS_dir: Optional[str|os.PathLike] = None
    MNIST_Pneumonia_dir: Optional[str|os.PathLike] = None
    MNIST_Retina_dir: Optional[str|os.PathLike] = None
    MNIST_Tissue_dir: Optional[str|os.PathLike] = None
    DiffVQA_dir: Optional[str|os.PathLike] = None
    CheXBert_dir:Optional[str|os.PathLike] = None
    physionet_username: Optional[str] = None
    physionet_password: Optional[str] = None
    device: Optional[str] = "cuda"

    def __post_init__(self):
        if self.device != "cpu":
            # Check if the device is available (handle cuda and mps)
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"

            elif self.device == "mps" and not torch.backends.mps.is_available():
                self.device = "cpu"


def remove_punctuation(input_string: str):
    # Make a translator object to replace punctuation with none
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    # Use the translator
    return input_string.translate(translator)


def csvWriter(data, path):
    try:
        with open(f"{path}.csv", "w", newline="") as f:
            spamWriter = csv.writer(f)
            spamWriter.writerows(data)
    except Exception as e:
        print(e)


def jsonWriter(data, path):
    try:
        with open(f"{path}.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)


SUPPORTED_FILETYPES = {"csv": csvWriter, "json": jsonWriter}


def fileWriterFactory(fileType):
    assert fileType in SUPPORTED_FILETYPES, f"{fileType} not supported."

    return SUPPORTED_FILETYPES[fileType]


def exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list):
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
        sum([1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list)])
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum([1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list)])
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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

    sections = list()
    section_names = list()
    section_idx = list()

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

    section_names = normalize_section_names(section_names)

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


def normalize_section_names(section_names):
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
    p_findings = re.compile("({})".format("|".join(p_findings)))

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


def custom_mimic_cxr_rules():
    custom_section_names = {
        "s50913680": "recommendations",  # files/p11/p11851243/s50913680.txt
        "s59363654": "examination",  # files/p12/p12128253/s59363654.txt
        "s59279892": "technique",  # files/p13/p13150370/s59279892.txt
        "s59768032": "recommendations",  # files/p13/p13249077/s59768032.txt
        "s57936451": "indication",  # files/p14/p14325424/s57936451.txt
        "s50058765": "indication",  # files/p14/p14731346/s50058765.txt
        "s53356173": "examination",  # files/p15/p15898350/s53356173.txt
        "s53202765": "technique",  # files/p16/p16076182/s53202765.txt
        "s50808053": "technique",  # files/p16/p16631485/s50808053.txt
        "s51966317": "indication",  # files/p10/p10817099/s51966317.txt
        "s50743547": "examination",  # files/p11/p11388341/s50743547.txt
        "s56451190": "note",  # files/p11/p11842879/s56451190.txt
        "s59067458": "recommendations",  # files/p11/p11984647/s59067458.txt
        "s59215320": "examination",  # files/p12/p12408912/s59215320.txt
        "s55124749": "indication",  # files/p12/p12428492/s55124749.txt
        "s54365831": "indication",  # files/p13/p13876470/s54365831.txt
        "s59087630": "recommendations",  # files/p14/p14267880/s59087630.txt
        "s58157373": "recommendations",  # files/p15/p15032392/s58157373.txt
        "s56482935": "recommendations",  # files/p15/p15388421/s56482935.txt
        "s58375018": "recommendations",  # files/p15/p15505556/s58375018.txt
        "s54654948": "indication",  # files/p17/p17090359/s54654948.txt
        "s55157853": "examination",  # files/p18/p18975498/s55157853.txt
        "s51491012": "history",  # files/p19/p19314266/s51491012.txt
    }

    custom_indices = {
        "s50525523": [201, 349],  # files/p10/p10602608/s50525523.txt
        "s57564132": [233, 554],  # files/p10/p10637168/s57564132.txt
        "s59982525": [313, 717],  # files/p11/p11989982/s59982525.txt
        "s53488209": [149, 475],  # files/p12/p12458657/s53488209.txt
        "s54875119": [234, 988],  # files/p13/p13687044/s54875119.txt
        "s50196495": [59, 399],  # files/p13/p13894879/s50196495.txt
        "s56579911": [59, 218],  # files/p15/p15394326/s56579911.txt
        "s52648681": [292, 631],  # files/p15/p15666238/s52648681.txt
        "s59889364": [172, 453],  # files/p15/p15835529/s59889364.txt
        "s53514462": [73, 377],  # files/p16/p16297706/s53514462.txt
        "s59505494": [59, 450],  # files/p16/p16730991/s59505494.txt
        "s53182247": [59, 412],  # files/p16/p16770442/s53182247.txt
        "s51410602": [47, 320],  # files/p17/p17069955/s51410602.txt
        "s56412866": [522, 822],  # files/p17/p17612000/s56412866.txt
        "s54986978": [59, 306],  # files/p17/p17912487/s54986978.txt
        "s59003148": [262, 505],  # files/p17/p17916384/s59003148.txt
        "s57150433": [61, 394],  # files/p18/p18335791/s57150433.txt
        "s56760320": [219, 457],  # files/p18/p18418794/s56760320.txt
        "s59562049": [158, 348],  # files/p18/p18502016/s59562049.txt
        "s52674888": [145, 296],  # files/p19/p19381919/s52674888.txt
        "s55258338": [192, 568],  # files/p13/p13719117/s55258338.txt
        "s59330497": [140, 655],  # files/p15/p15479218/s59330497.txt
        "s52119491": [179, 454],  # files/p17/p17959278/s52119491.txt
        # below have no findings at all in the entire report
        "s58235663": [0, 0],  # files/p11/p11573679/s58235663.txt
        "s50798377": [0, 0],  # files/p12/p12632853/s50798377.txt
        "s54168089": [0, 0],  # files/p14/p14463099/s54168089.txt
        "s53071062": [0, 0],  # files/p15/p15774521/s53071062.txt
        "s56724958": [0, 0],  # files/p16/p16175671/s56724958.txt
        "s54231141": [0, 0],  # files/p16/p16312859/s54231141.txt
        "s53607029": [0, 0],  # files/p17/p17603668/s53607029.txt
        "s52035334": [0, 0],  # files/p19/p19349312/s52035334.txt
    }

    return custom_section_names, custom_indices


# def cleanStr(text: str):
#     tempStr = remove_punctuation(text.lower().replace("\n", " ").strip())
#     return re.sub(" +", " ", tempStr)



def download_file(url: str, fname: str, username=None, password=None):
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
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


import re

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
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
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


def cleanStr(token):
    token = token.lower()
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
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