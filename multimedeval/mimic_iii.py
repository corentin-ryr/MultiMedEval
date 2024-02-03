from multimedeval.utils import Benchmark, EvalParams
import os
import pandas as pd
from multimedeval.tqdm_loggable import tqdm_logging
import datasets
import re
from multimedeval.utils import Benchmark, exact_entity_token_if_rel_exists_reward
import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from torch.utils.data import DataLoader
from multimedeval.utils import download_file
import gzip
import shutil
from multimedeval.mimic import compute_bertscore, compute_meteor, compute_composite


def get_final_report(text):
    if "FINAL REPORT" not in text:
        return None
    idx = text.index("FINAL REPORT")
    text = text[idx:]
    while "(Over)" in text and "(Cont)" in text:
        text = text[0 : text.index("(Over)")] + text[text.index("(Cont)") + 6 :]
    return text


def extract_sections(text):
    p_section = re.compile(r"\n ([A-Z ()/,-]+):\s", re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)
    if s:
        sections.append(text[12 : s.start(1)])
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

    preprocessed = [sec.strip().lower() for sec in sections]
    preprocessed = [re.sub("\n", "", sec) for sec in preprocessed]
    preprocessed = [re.sub(" +", " ", sec) for sec in preprocessed]
    return (section_names, preprocessed)


class MIMIC_III(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = "MIMIC-III"
        self.modality = "Radiology"
        self.task = "Report Summarization"

        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys="rougeL")

    def setup(self):
        self.path = self.engine.getConfig()["MIMIC_III_dir"]

        self._generate_dataset()

        # reports_csv = pd.read_csv(os.path.join(self.path, "NOTEEVENTS.csv"), low_memory=False)
        # reports_csv = reports_csv.fillna(-1)

        reports_csv = []
        # Open the NOTEEVENTS.csv file and keep the reports that are in the mapping in the reports_csv list
        reports_csv = pd.read_csv(os.path.join(self.path, "NOTEEVENTS.csv"), low_memory=False)
        reports_csv = reports_csv.fillna(-1)

        expToReport = {}
        for EXP in tqdm_logging(self.logger, mapping.keys(), desc="Extracting reports"):
            filter_reports = reports_csv[reports_csv["DESCRIPTION"].isin(mapping[EXP])]
            reports_list = filter_reports["TEXT"].tolist()
            reports_ids = filter_reports["ROW_ID"].tolist()
            missing_idx = []
            all_sections = []
            impressions_list = []
            findings_list = []
            ids_list = []
            for i in range(len(reports_list)):
                report = reports_list[i]
                text = get_final_report(report)
                # No reports ? we skip
                if text is None:
                    missing_idx.append(reports_list.index(report))
                    continue

                # Getting all sections from the reports
                section_names, sections = extract_sections(text)
                for j in range(len(section_names)):
                    if section_names[j] in section_map_rev:
                        section_names[j] = section_map_rev[section_names[j]]
                all_sections.extend(section_names)

                # Is there no or two impressions ? Its safer to skip (multiple studies of differents body parts in the same reports)
                count = section_names.count("impression")
                if count > 1 or count == 0:
                    continue

                # Finding the findings
                impression_text = sections[section_names.index("impression")]
                section_names.remove("impression")
                findings_text = ""
                for m in findings_mapping[EXP]:
                    if m[0] in section_names:
                        findings_text = sections[section_names.index(m[0])]
                        if findings_text:
                            break

                # No findings ? Skip
                if not findings_text or not impression_text:
                    continue

                findings_list.append(re.sub("\s+", " ", findings_text))
                impressions_list.append(re.sub("\s+", " ", impression_text))
                ids_list.append(reports_ids[i])

            # preprocessing the findings and impression.
            findings_list_clean = []
            impressions_list_clean = []
            for f in findings_list:
                for replace in re.findall(r"\[\*\*(.*?)\*\*\]", f):
                    f = f.replace("[**{}**]".format(replace), "___")
                findings_list_clean.append(f)

            for f in impressions_list:
                for replace in re.findall(r"\[\*\*(.*?)\*\*\]", f):
                    f = f.replace("[**{}**]".format(replace), "___")
                impressions_list_clean.append(f)

            assert (len(impressions_list_clean)) == (len(findings_list_clean))

            expToReport[EXP] = {"impression": impressions_list_clean, "findings": findings_list_clean, "ids": ids_list}

        # Open the split csv
        split = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mimiciiisplit.csv"))
        split = split[split["split"] == "test"]
        # Convert to a list of ids
        split = split["ids"].tolist()

        # Get all the ids for the test set from the extToReport dict
        datasetTest = []
        for folder in expToReport.keys():
            ids = expToReport[folder]["ids"]
            reports = expToReport[folder]["findings"]
            impression = expToReport[folder]["impression"]
            for i in range(len(ids)):
                if ids[i] in split:
                    datasetTest.append(
                        {
                            "findings": reports[i],
                            "impression": impression[i],
                            "split": "test",
                            "modality": folder.split("_")[0],
                            "anatomy": folder.split("_")[1],
                        }
                    )

        self.dataset = datasets.Dataset.from_list(datasetTest)

    def run(self, params: EvalParams, batcher):
        self.logger.info(f"***** Benchmarking : {self.taskName} *****")
        refReports = []
        hypReports = []
        bleu1Scores = []
        bleu4Scores = []
        rougeLScores = []

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

            for hyp, ref in zip(batcherHyp, batcherCorrect):
                bleu1Scores.append(self.bleu_1([hyp], [[ref]]).item())
                bleu4Scores.append(self.bleu_4([hyp], [[ref]]).item())
                rougeLScores.append(self.rougeL([hyp], [[ref]])["rougeL_fmeasure"].item())

        f1_bertscore = compute_bertscore(hypReports, refReports)
        f1_bertscore_unscaled = compute_bertscore(hypReports, refReports, rescale=False)

        chexbert_similarity = self.compute_chexbert(hypReports, refReports)

        f1_radgraph = self.compute_radgraph(hypReports, refReports)

        bleu_scores = torch.tensor(
            [self.bleu_1([candidate], [[reference]]).item() for reference, candidate in zip(refReports, hypReports)]
        )

        radcliq_v0_scores = compute_composite(bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph)

        meteor_scores = compute_meteor(hypReports, refReports)

        rougeScores = self.rougeL.compute()
        rougeScores = {key: value.item() for key, value in rougeScores.items()}

        metrics = {
            "bleu1": self.bleu_1.compute().item(),
            "bleu4": self.bleu_4.compute().item(),
            "f1-radgraph": f1_radgraph.mean().item(),
            "CheXBert vector similarity": chexbert_similarity.mean().item(),
            "f1-bertscore": f1_bertscore_unscaled.mean().item(),
            "radcliq": sum(radcliq_v0_scores) / len(radcliq_v0_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
        }
        metrics.update(rougeScores)

        answersLog = zip(refReports, hypReports, bleu1Scores, bleu4Scores, rougeLScores)
        # Add a header to the log
        answersLog = [("ref", "hyp", "bleu1", "bleu4", "rougeL")] + list(answersLog)

        return [
            {"type": "json", "name": f"metrics_{self.taskName}", "value": metrics},
            {"type": "csv", "name": self.taskName, "value": answersLog},
        ]

    def getCorrectAnswer(self, sample, fullText=False):
        return sample["impression"]

    def format_question(self, sample):
        question = sample["findings"]
        question += "\nSummarize the findings."
        formattedText = [
            {
                "role": "user",
                "content": question,
            }
        ]
        return (formattedText, [])

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

    def _generate_dataset(self):
        # Check if the path already exists and if so return
        if os.path.exists(os.path.join(self.path, "physionet.org", "files", "mimiciii", "1.4", "NOTEEVENTS.csv")):
            self.path = os.path.join(self.path, "physionet.org", "files", "mimiciii", "1.4")
            return

        os.makedirs(os.path.join(self.path, "physionet.org", "files", "mimiciii", "1.4"), exist_ok=True)

        username, password = self.engine.getPhysioNetCredentials()
        # wget_command = f'wget -r -N -c -np --directory-prefix "{self.path}" --user "{username}" --password "{password}" https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz'
        # subprocess.run(wget_command, shell=True, check=True)

        download_file(
            "https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz",
            os.path.join(self.path, "physionet.org", "files", "mimiciii", "1.4", "NOTEEVENTS.csv.gz"),
            username,
            password,
        )

        self.path = os.path.join(self.path, "physionet.org", "files", "mimiciii", "1.4")

        # Unzip the NOTEEVENTS file
        file = os.path.join(self.path, "NOTEEVENTS.csv")
        with gzip.open(file + ".gz", "rb") as f_in:
            with open(file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the zip file
        os.remove(file + ".gz")


# which reports go into the given modality_anatomy pair

mapping = {
    "CT_head": [
        "CT HEAD W/O CONTRAST",
        "CT EMERGENCY HEAD W/O CONTRAST",
        "CT HEAD W/ CONTRAST",
        "CT HEAD W/ & W/O CONTRAST",
        "CTA HEAD W&W/O C & RECONS",
        "CT HEAD W/ ANESTHESIA W/ CONTRAST",
        "CT HEAD W/ ANESTHESIA W/O CONTRAST",
        "PORTABLE HEAD CT W/O CONTRAST",
        "CT BRAIN PERFUSION",
    ],
    "CT_neck": [
        "CT NECK W/CONTRAST (EG:PAROTIDS)",
        "CT NECK W/O CONTRAST (EG: PAROTIDS)",
        "CT NECK W/CONT +RECONSTRUCTION",
        "CT NECK W/ & W/O CONTRAST",
        "CTA NECK W&W/OC & RECONS",
        "CT NECK W/CONTRAST W/ONC TABLES",
    ],
    "CT_sinus": [
        "CT SINUS/MAXLIOFACIAL W/O CONTRAST",
        "CT SINUS W/ CONTRAST",
        "CT SINUS/MANDIBLE/MAXILLOFACIAL W/O CONTRAST",
        "CT SINUS CORONAL ONLY",
        "CT SINUS AXIAL W/VTI",
        "CT SINUS W/ & W/O CONTRAST",
        "CT SINUS W/O CONTRAST FOR SURGICAL PLANNING",
        "CT SINUS/MANDIBLE/MAXILLOFACIAL W/ CONTRAST",
        "CT SINUS/MANDIBLE/MAXILLOFACIAL W & W/O CONTRAST",
    ],
    "CT_spine": [
        "CT C-SPINE W/O CONTRAST",
        "CT C-SPINE W/CONTRAST",
        "CT T-SPINE W/O CONTRAST",
        "CT T-SPINE W/ CONTRAST",
        "CT L-SPINE W/O CONTRAST",
        "CT L-SPINE W/ CONTRAST",
    ],
    "CT_chest": [
        "CT CHEST W/CONTRAST",
        "CHEST CTA WITH CONTRAST",
        "CT CHEST W/CONT+RECONSTRUCTION",
        "CT CHEST W&W/O C",
        "CTA CHEST W&W/O C &RECONS",
        "CT STEREOTAXIS CHEST W/ CONTRAST",
        "CTA CHEST W&W/O C&RECONS, NON-CORONARY",
        "CT CHEST W/CONTRAST W/ONC TABLE",
        "CT CHEST W/O CONTRAST W/ONC TABLES",
        "P CTA CHEST W&W/O C&RECONS, NON-CORONARY PORT",
        "CT CHEST AND ABDOMEN W/O CONTRAST",
    ],
    "CT_abdomen-pelvis": [
        "CT ABDOMEN W/O CONTRAST",
        "CT ABDOMEN W/CONTRAST",
        "CT ABDOMEN W & W/O CONTRAST W/ONC TABLES",
        "CT ABDOMEN W/CONTRAST W/ONC TABLE",
        "CT ABDOMEN W/O CONTRAST W/ONC TABLE",
        "CT ABDOMEN AND PELVIS W/O CONTRAST W/ONC TABLES",
        "CT PELVIS W/CONTRAST",
        "CT PELVIS W/O CONTRAST",
        "CT PELVIS W&W/O C",
        "CTA PELVIS W&W/O C & RECONS",
        "CT ABD & PELVIS",
        "CT ABD&PELVIS W/O C COLON TECHNIQUE",
        "CT ABD&PELVIS W&W/O C COLON TECHNIQUE",
        "CT ABD&PELVIS W/C COLON TECHNIQUE",
        "CT PELVIS W/CONTRAST W/ONC TABLES",
        "CT PELVIS W/O CONTRAST W/ONC TABLES",
        "CT ABD & PELVIS WITH CONTRAST",
        "CT ABD & PELVIS W/O CONTRAST",
        "CT ABD & PELVIS W & W/O CONTRAST, ADDL SECTIONS",
        "P CT ABD & PELVIS W & W/O CONTRAST, ADDL SECTIONS PORT",
        "CTA ABD & PELVIS",
        "CT ABDOMEN AND PELVIS W/O CONTRAST W/ONC TABLES",
    ],
    "MR_abdomen": [
        "MR ABDOMEN W/ CONTRAST",
        "MR ABDOMEN",
        "MR ABDOMEN W&W/O CONTRAST",
        "MR ABDOMEN W/O CONTRAST",
        "MR ABDOMEN W/CONTRAST",
        "MRI ABDOMEN W/O & W/CONTRAST",
        "MRA ABDOMEN W&W/O CONTRAST",
        "MRI ABDOMEN W/O CONTRAST",
        "MRA ABDOMEN W/O CONTRAST",
        "MRV ABDOMEN W/O CONTRAST",
        "MRV ABDOMEN W&W/O CONTRAST",
    ],
    "MR_pelvis": [
        "MR PELVIS W & W/O CONTRAST",
        "MRA PELVIS",
        "MRI PELVIS WITHOUT CONTRAST",
        "MR PELVIS W/ CONTRAST",
        "MR PELVIS W/O CONTRAST",
        "MRI PELVIS W/O & W/CONTRAST",
        "MRI PELVIS W/O CONTRAST",
        "MRA PELVIS W&W/O CONTRAST",
        "MRA PELVIS W/O CONTRAST",
        "MRV PELVIS W&W/O CONTRAST",
        "MRV PELVIS W/O CONTRAST",
        "MR PELVIS W&W/O CONTRAST",
        "MR GYN PELVIS  W & W/O CONTRAST",
        "MRI PELVIS W/CONTRAST",
        "MR GYN PELVIS W&W/O CONTRAST",
    ],
    "MR_spine": [
        "MR L SPINE SCAN",
        "MR L-SPINE W & W/O CONTRAST",
        "MR L SPINE WITH CONTRAST",
        "MR L SPINE W/O CONTRAST",
        "MRA LUMBAR SPINE",
        "MR CERVICAL SPINE",
        "MR C-SPINE SCAN WITH CONTRAST",
        "MR C-SPINE W& W/O CONTRAST",
        "MR CERVICAL SPINE W/O CONTRAST",
        "MRA CERVICAL SPINE",
    ],
    "MR_head": [
        "MR HEAD W & W/O CONTRAST",
        "MR HEAD NEURO",
        "MR HEAD W/ CONTRAST",
        "MR-ANGIO HEAD W/OUT CONTRAST",
        "MR-ANGIO HEAD",
        "MR HEAD W/O CONTRAST",
        "MR-ANGIO HEAD W/ CONTRAST",
        "MR-ANGIO HEAD W & W/O CONTRAST",
        "MRV HEAD W/O CONTRAST",
        "MR HEAD W/CNTRST&TUMOR VOLUMETRIC",
        "MR HEAD W&W/OC FOR PTS W/ DBS",
    ],
    "MR_neck": [
        "MR S.T. NECK W & W/O GADO",
        "MR-ANGIO NECK WITHOUT CONTRAST",
        "MR NECK W/O CONTRAST",
        "MR-ANGIO NECK W & W/O CONTRAST",
        "MRA NECK W&W/O CONTRAST",
        "MRI SOFT TISSUE NECK, W/O & W/CONTRAST",
        "MR ANGIOGRAM NECK, W/O & W/CONTRAST",
        "MR ANGIOGRAM NECK, W/O CONTRAST",
        "MRI SOFT TISUUE NECK, W/O CONTRAST",
        "MRA NECK W/CONTRAST",
        "MRA NECK W/O CONTRAST",
        "MRV NECK W/O CONTRAST",
    ],
}

# Typos
section_map = {
    "findings": [
        "findigns",
        "findigs",
        "finding",
        "findings",
        "findnings",
        "fidings",
        "findings",
        "findings - brain mri",
        "findings and impression",
        "findings brain mri",
        "findings mri of the brain",
        "findings, brain mri",
        "findings-brain mri",
        "findings-mri of the head",
        "findings",
        "findings ct chest",
        "findings for ct of the chest",
        "findings head ct",
        "findngs",
        "findings ct head",
        "findins",
        "findnigs",
        "finidngs",
        "findings",
        "findgins",
        "findgings",
    ],
    "impression": [
        "impession",
        "impresiion",
        "impresion",
        "impression",
        "impression and plan",
        "impressions",
        "impresssion",
        "imprssion",
        "impression for mri of the brain",
        "impression of mri of the brain",
        "impression of the mra",
        "imrpession",
        "impresison",
        "impressiion",
        "impression",
        "imression",
        "imprression",
        "impresson",
        "impreesion",
        "imppression",
        "impression",
    ],
    "technique": [
        "techinique",
        "techinque",
        "techique",
        "techniqe",
        "technique",
        "technique and procedure",
        "technique contrast-enhanced",
        "technique of study",
        "techniques",
        "technque",
        "techniqu",
        "technique",
        "technique for mra of the head",
        "technnique",
        "technqiue",
        "tecnique",
        "tehnique",
        "techniique",
        "techniqur",
        "technique head ct",
        "tecnhique",
        "techniqie",
        "techniquie",
        "tecnhinque",
    ],
    "indication": [
        " indication",
        "indicaation",
        "indicaiton",
        "indication",
        "indication ",
        "indication for exam",
        "indication for study",
        "indication for the study",
        "indication of study",
        "indication of the study",
        "indication of the test",
        "indications",
        "indicdation",
        "indcation",
        "indicatioin",
        "indication",
        "indication for the exam",
        "indicaton",
        "indication",
        "indicatton",
        "indiciaton",
        "indiction",
        "indiation",
        "indicatiion",
    ],
    "comparison": [
        "comaprison",
        "comparions",
        "comparision",
        "comparisions",
        "comparison",
        "comparison ",
        "comparison available",
        "comparison exam",
        "comparison exams",
        "comparison studies",
        "comparison study",
        "comparisonm",
        "comparisono",
        "comparisons",
        "comparisons available",
        "comparisson",
        "comparsion",
        "comparsions",
        "complications",
        "comprison",
        "comparison",
        "comparison  study",
        "comparison examination",
        "comparison examinations",
        "comparison mr study",
        "comparisosn",
        "comparson",
        "comparioson",
        "comparpison",
        "comparsison",
        "comprarison",
        "comparisoin",
        "comparisoni",
        "comparion",
        "comparison with",
        "comaparison",
    ],
    "bone windows": [
        "bond windows",
        "bone  windows",
        "bone dindows",
        "bone window",
        "bone windows",
        "bone windowss",
        "bone winodws",
        "bone winoows",
        "bones windows",
        "bony window",
        "bony windows",
        "bone eindows",
    ],
    "osseous structures": [
        "osseous structions",
        "osseous structure",
        "osseous structures",
        "osseous structurs",
        "osseous strucures",
        "osseus structures",
        "osseous structures",
        "osseous structures ",
        "ossseous structures",
    ],
    "clinical information": [
        "clincal information",
        "clinical  information",
        "clinical indication",
        "clinical information",
        "clinincal information",
        "clnical information",
        "cliical information",
        "clinucal information",
    ],
    "exam": ["exam", "examination"],
    "non-contrast head ct": [
        "non-contrast head ct",
        "noncontrast head ct",
        "non-contrast head ct scan",
        "non-contrast ct head",
        "noncontrast head ct scan",
        "non-contrast ct of the head",
        "noncontrast ct head",
        "noncontrast ct of the head",
        "non-contrast ct scan of the head",
        "non contrast head ct",
        "non-contrast head",
        "non-contrast ct",
        "noncontrast ct scan of the head",
        "non contrast head ct scan",
        "noncontrast  head ct",
        "noncontrast ct scan",
        "non-contrast ct of head",
        "noncontrast ct",
        "non-contrast  head ct",
        "non-contrasted head ct scan",
        "non-contrast head ct head",
        "non-contrast head  ct",
        "non-enhanced ct of the head",
        "non-enhanced head ct",
        "non-contrast head ct scan findings",
        "non contrast ct of the head",
        "non contrast ct findings",
        "non-contrast-enhanced ct scan",
        "noncontrast head",
        "noncontrast enhanced ct",
        "non-contrasted head ct",
        "non-contrasat head ct",
        "non-conrast head ct",
        "non contrast ct head",
    ],
    "ct of the abdomen without iv contrast": [
        "ct abdomen w/o iv contrast",
        "ct abdomen with no iv contrast",
        "abdomen ct w/o iv contrast",
        "abdomen ct without iv contrast",
        "ct abdomen w/o iv contast",
        "ct of the abdomen without iv constrast",
        "ct of the abdomen without iv contrast",
        "ct of the abdomen without iv contrast administration" "ct abdomen without iv contrast",
        "ct of the abdomen without iv contrast",
        "ct of the abdomen with no iv contrast",
        "ct of the abdomen with no iv contrast administration",
        "abdomen without iv contrast",
        "ct abdomen without iv contrast findings",
        "ct abdomen, without iv contrast",
    ],
    "abdomen": ["abdomen ", "adbomen", "abodmen", "ct abdmone", "ct abdomen", "ct abdomen "],
    "indication": [" indication"],
    "subcutaneous tissues": ["subcutaneous tissue"],
    "comment": [" comment"],
    "angiogram": ["angiogram)"],
    "ct of the abdomen with iv contrast": [
        "abdomen ct with iv contrast",
        "abdomen with iv contrast",
        "cc of the abdomen with iv contrast",
        "ct abdomen with iv contrast",
        "ct abdomen with iv contrast findings",
        "ct abdomen with iv contrast only",
        "ct of abdomen with iv contrast",
        "ct of the  abdomen with iv contrast",
        "ct of the abdomen and with iv contrast",
        "ct of the abdomen with iv contrast",
        "ct of the abdomen with iv contrast only",
        "ct of the abdomen with iv contrast technique",
        "ct of the abdomen with with iv contrast",
        "ct of the abdoment with iv contrast",
        "ct with iv contrast of the abdomen",
        "cta abdomen with iv contrast",
        "findings ct of the abdomen with iv contrast",
        "findings for ct of the abdomen with iv contrast",
        "abdomen ct w/ iv contrast",
        "abdomen ct w/iv contrast",
        "ct abdomen w/ iv contrast",
        "ct abdomen w/iv contrast",
        "ct of abdomen w/ iv contrast",
        "ct of abdomen w/iv contrast",
        "ct of the abdomen w/iv contrast",
        "ct abdomen w/intravenous contrast",
        "abdomen ct with intravenous contrast",
        "ct abdomen with intravenous contrast",
        "ct of abdomen with intravenous contrast",
        "ct of the abdomen with intravenous contrast",
        "ct of the abdomen with intravenous contrast only",
        "ct of the abdomen with the administration of intravenous contrast",
        "ct of the abdomen, with intravenous contrast",
        "ct of the abdoment with intravenous contrast",
        "ct of the the abdomen with intravenous contrast",
        "ct scan abdomen with intravenous contrast",
        "ct scan of abdomen (with intravenous contrast)",
        "ct scan of abdomen with intravenous contrast",
        "ct scan of the abdomen (with intravenous contrast)",
        "ct scan of the abdomen with intravenous contrast",
        "cta abdomen with intravenous contrast",
        "abdominal ct with iv contrast",
        "abdominal ct with intravenous contrast",
        "abdominal ct with iv contrast findings",
        "abdominal ct with iv contrast, findings",
        "ct of the abdominal with iv contrast",
    ],
    "ct of the abdomen without contrast": [
        "abdomen ct without contrast",
        "abdomen without contrast",
        "ct abdomen without contrast",
        "ct abdomen without contrast ",
        "ct abdomen, limited without contrast",
        "ct abdomen, without contrast",
        "ct of abdomen without contrast",
        "ct of the abdomen without additional contrast",
        "ct of the abdomen without and without contrast",
        "ct of the abdomen without contrast",
        "ct of the abdomen without contrast, findings",
        "ct of the abdomen/pelvis without contrast",
        "ct scan of the abdomen without contrast",
        "findings for ct of the abdomen without contrast",
        "findings, abdomen without contrast",
        "preliminary ct of the abdomen without contrast",
    ],
    "ct of the chest without contrast": [
        "chest ct without contrast",
        "ct chest without contrast",
        "ct of the chest without contrast",
        "ct of the chest without contrast, findings",
        "ct scan of the chest without contrast",
        "findings for ct of the chest without contrast",
        "ct chest w/o contrast",
        "ct of the chest w/o contrast",
        "limited ct chest w/o contrast",
    ],
    "ct of the chest without iv contrast": [
        "chest ct w/o iv contrast",
        "ct chest w/o iv contrast",
        "chest ct without iv contrast",
        "ct chest without iv contrast",
        "ct of chest without iv contrast",
        "ct of lower chest without iv contrast",
        "ct of the chest without iv contrast",
        "chest ct without intravenous contrast",
        "chest without intravenous contrast",
        "ct chest without intravenous contrast",
        "ct of the chest without intravenous contrast",
    ],
    "ct of the chest with iv contrast": [
        "chest ct with intravenous contrast",
        "ct chest with intravenous contrast",
        "ct of chest with intravenous contrast",
        "ct of the chest with intravenous contrast",
        "cta chest with intravenous contrast",
        "cta of the chest with intravenous contrast",
    ],
    "ct of the pelvis without contrast": [
        "ct of pelvis without contrast",
        "ct of the pelvis without contrast",
        "ct of the pelvis without contrast, findings",
        "ct pelvis without contrast",
        "ct pelvis without contrast findings",
        "ct pelvis, without contrast",
        "ct scan of the pelvis without contrast",
        "ctof the pelvis without contrast",
        "findings for pelvis without contrast",
        "pelvis ct without contrast",
        "pelvis without contrast",
        "ct of pelvis w/o contrast",
        "ct of the pelvis w/o contrast",
        "ct pelvis w/o contrast",
        "ct pelvis w/o contrast ",
        "pelvis ct w/o contrast",
    ],
    "ct of the pelvis with iv contrast": [
        "ct of pelvis with intravenous contrast",
        "ct of te pelvis with intravenous contrast",
        "ct of the of the pelvis with intravenous contrast",
        "ct of the pelvis with intravenous contrast",
        "ct of the pelvis with intravenous contrast only",
        "ct pelvis with intravenous contrast",
        "ct scan of pelvis with intravenous contrast",
        "ct scan of the pelvis with intravenous contrast",
        "ct scan pelvis with intravenous contrast",
        "ct the pelvis with intravenous contrast",
        "pelvis ct with intravenous contrast",
        "pelvis with intravenous contrast",
        "ct  pelvis with iv contrast",
        "ct of pelvis with iv contrast",
        "ct of pelvis with iv contrast only",
        "ct of the pelvis with iv contrast",
        "ct of the pelvis with iv contrast only",
        "ct of the pelvis with only iv contrast",
        "ct pelvis with iv contrast",
        "ct pelvis with iv contrast findings",
        "ct pelvis with iv contrast only",
        "ct pelvis with no iv contrast",
        "ct scan of pelvis with iv contrast",
        "ct scan of the pelvis with iv contrast",
        "ct the pelvis with iv contrast",
        "cta pelvis with iv contrast",
        "pelvis ct with iv contrast",
        "pelvis with iv contrast",
    ],
}

# Which sections are considered findings for the given modality_anatomy pair (with frequency)
findings_mapping = {
    "CT_head": [
        ("findings", 26640),
        ("non-contrast head ct", 3325),
        ("ct head", 731),
        ("ct head without contrast", 695),
        ("ct head without iv contrast", 630),
        ("head ct", 619),
        ("head ct without iv contrast", 516),
        ("cta head", 416),
        ("head ct without contrast", 375),
        ("ct of the head without contrast", 292),
        ("ct perfusion", 267),
        ("ct head w/o contrast", 264),
        ("head cta", 199),
        ("cta", 191),
        ("ct of the head without iv contrast", 151),
        ("cta neck", 143),
        ("head and neck cta", 141),
        ("ct of the brain without intravenous contrast", 129),
        ("cta of the head", 114),
        ("ct angiography of the head", 98),
        ("cta head and neck", 84),
        ("ct angiogram of the head", 78),
        ("ct of the head", 69),
        ("ct angiography head", 69),
        ("ct head findings", 64),
        ("ct head w/o iv contrast", 63),
        ("ct scan of the brain", 61),
        ("head ct without intravenous contrast", 57),
        ("ct head without intravenous contrast", 52),
        ("ct angiogram", 50),
        ("cta of the head and neck", 45),
        ("head ct w/o contrast", 41),
        ("ct of the brain without iv contrast", 39),
        ("head ct w/o iv contrast", 38),
        ("ct angiography", 36),
        ("ct angiogram of the head and neck", 34),
        ("ct scan of the head without contrast", 30),
        ("ct of head without iv contrast", 30),
        ("ct brain without contrast", 30),
        ("ct head without and with contrast", 28),
        ("ct of the head without intravenous contrast", 28),
        ("ct of the head w/o contrast", 27),
        ("ct", 27),
        ("ct of the head without and with iv contrast", 25),
        ("ct of the head without and with contrast", 25),
        ("ct head with no iv contrast", 18),
        ("ct brain w/o iv contrast", 18),
        ("ct head without and with iv contrast", 18),
        ("ct reconstructions", 16),
        ("ct of the brain without and with intravenous contrast", 14),
        ("ct scan of the head", 13),
        ("ct of head without contrast", 12),
        ("ct angiogram of the neck", 11),
        ("ct head with iv contrast", 9),
        ("ct head with no contrast", 9),
        ("ct head with and without contrast", 8),
        ("head ct with iv contrast", 8),
        ("ct head with contrast", 8),
        ("ct of the brain without contrast", 8),
        ("cta of the brain", 8),
        ("cta findings", 8),
        ("ct of the head with contrast", 7),
        ("ct of the head w/o iv contrast", 7),
        ("ct head before and after iv contrast", 7),
        ("contrast head ct", 7),
        ("emergency head ct scan", 7),
        ("ct of the head without and with intravenous contrast", 7),
        ("ct brain", 7),
        ("cta brain", 7),
        ("ct of the head with and without iv contrast", 6),
        ("ct head with and without iv contrast", 6),
        ("ct brain without iv contrast", 6),
        ("cta brain findings", 6),
    ],
    "CT_neck": [
        ("findings", 1068),
        ("ct of the neck with iv contrast", 43),
        ("cta neck", 32),
        ("ct neck with iv contrast", 27),
        ("cta of the neck", 23),
        ("cta", 21),
        ("ct neck", 19),
        ("ct neck with contrast", 18),
        ("ct of the neck with contrast", 13),
        ("preliminary report", 13),
        ("ct of the neck", 12),
        ("neck cta", 11),
        ("ct of the neck with intravenous contrast", 11),
        ("ct angiography of the neck", 11),
        ("neck ct with iv contrast", 9),
        ("ct of the neck without iv contrast", 8),
        ("ct angiography neck", 7),
        ("ct of the neck without contrast", 5),
        ("ct neck without contrast", 5),
        ("ct neck without iv contrast", 5),
        ("neck ct with contrast", 5),
        ("ct of the neck without intravenous contrast", 5),
        ("ct neck with intravenous contrast", 5),
    ],
    "CT_sinus": [
        ("findings", 1226),
        ("sinus ct", 50),
        ("ct sinus", 31),
        ("ct sinuses", 15),
        ("ct of the sinuses", 9),
        ("ct of the paranasal sinuses", 6),
        ("non-contrast sinus ct", 6),
        ("ct sinus/maxillofacial", 5),
        ("sinus ct without contrast", 4),
        ("ct of the paranasal sinuses without iv contrast", 4),
        ("ct sinus without contrast", 4),
        ("ct sinuses without contrast", 3),
        ("ct sinuses without iv contrast", 3),
        ("ct of the sinuses without contrast", 3),
        ("ct sinus/mandible/maxillofacial without contrast", 3),
        ("ct sinus/maxliofacial", 2),
        ("ct of the paranasal sinuses without contrast", 2),
    ],
    "CT_spine": [
        ("findings", 4725),
        ("ct c-spine", 247),
        ("ct of the cervical spine", 95),
        ("ct cervical spine", 81),
        ("ct reconstructions", 76),
        ("ct of the cervical spine without contrast", 48),
        ("cervical spine ct", 45),
        ("ct c-spine without iv contrast", 42),
        ("ct c-spine without contrast", 41),
        ("ct of the cervical spine without intravenous contrast", 33),
        ("c-spine ct without contrast", 32),
        ("ct of the cervical spine without iv contrast", 26),
        ("ct cervical spine w/o contrast", 26),
        ("ct cervical spine without contrast", 24),
        ("ct lumbar spine", 21),
        ("ct cervical spine without iv contrast", 20),
        ("ct t-spine", 18),
        ("ct thoracic spine", 17),
        ("non-contrast ct of the cervical spine with coronal and sagittal reformats", 17),
        ("ct of the thoracic spine", 16),
        ("thoracic spine", 16),
        ("ct c spine", 15),
        ("ct l-spine", 14),
        ("cervical spine", 14),
        ("lumbar spine", 13),
        ("non-contrast cervical spine ct", 13),
        ("prior study", 13),
        ("cervical spine ct without intravenous contrast", 13),
        ("ct cervical spine w/o iv contrast", 12),
        ("non-contrast ct c-spine", 12),
        ("noncontrast cervical spine ct", 11),
        ("ct c-spine findings", 11),
        ("ct c-spine without intravenous contrast", 10),
        ("ct of the thoracic spine without intravenous contrast", 10),
        ("ct of the lumbar spine", 9),
        ("ct of the lumbar spine without contrast", 9),
        ("ct of the c-spine", 9),
        ("ct thoracic spine without contrast", 9),
        ("non-contrast c-spine", 9),
        ("non-contrast ct of the cervical spine", 9),
        ("ct of the lumbar spine without iv contrast", 8),
        ("c-spine ct", 7),
        ("ct thoracic spine without iv contrast", 7),
        ("ct of the lumbar spine without intravenous contrast", 7),
        ("c spine ct without contrast", 6),
        ("ct of the thoracic spine without contrast", 6),
        ("cervical spine ct without contrast", 6),
        ("ct cervical spine without intravenous contrast", 6),
        ("ct lumbar spine without contrast", 5),
        ("ct scan of the cervical spine", 5),
        ("lumbar spine ct", 5),
        ("c-spine", 5),
        ("ct of the c-spine without iv contrast", 4),
        ("ct of the thoracic and lumbar spine", 4),
        ("ct c spine without iv contrast", 4),
        ("ct of the thoracic spine without iv contrast", 4),
        ("thoracic spine ct", 4),
        ("cervical spine ct scan", 4),
        ("ct l-spine without iv contrast", 4),
        ("ct of c-spine without iv contrast", 3),
        ("ct thoracic spine w/o contrast", 3),
        ("ct lumbar spine w/o contrast", 3),
        ("ct of c-spine", 3),
        ("ct lumbar spine without iv contrast", 3),
        ("ct of cervical spine without iv contrast", 3),
        ("ct thoracic spine with iv contrast", 3),
        ("ct of the cervical spine with sagittal and coronal reconstructions", 3),
        ("ct c-spine w/o iv contrast", 3),
        ("c-spine findings", 3),
        ("non-contrast cervical spine", 3),
        ("noncontrast ct c-spine", 3),
        ("ct spine", 3),
        ("ct lumbar spine without intravenous contrast", 3),
        ("ct c spine without contrast", 2),
        ("ct c-spine with reformations", 2),
    ],
    "CT_chest": [
        ("findings", 5478),
        ("ct of the chest with iv contrast", 2401),
        ("ct chest with iv contrast", 877),
        ("ct chest", 846),
        ("ct chest with contrast", 745),
        ("chest", 709),
        ("cta chest", 596),
        ("ct of the chest", 497),
        ("ct of the chest with contrast", 477),
        ("cta of the chest", 360),
        ("ct of the chest without and with iv contrast", 225),
        ("ct of the chest with and without iv contrast", 220),
        ("chest ct with iv contrast", 211),
        ("ct of the chest with and without intravenous contrast", 158),
        ("ct chest without and with iv contrast", 148),
        ("ct chest with and without intravenous contrast", 120),
        ("ct of the chest without and with intravenous contrast", 89),
        ("ct of the chest without and with contrast", 84),
        ("chest ct", 82),
        ("chest cta", 76),
        ("ct of chest with iv contrast", 75),
        ("ct angiogram chest", 71),
        ("chest with contrast", 55),
        ("chest ct angiogram", 52),
        ("cta chest with iv contrast", 49),
        ("chest ct with contrast", 47),
        ("ct chest before and after iv contrast", 44),
        ("ct chest with and without contrast", 42),
        ("ct of the chest with and without contrast", 40),
        ("ct of the chest with contrast, findings", 40),
        ("ct angiogram of the chest without and with intravenous contrast", 37),
        ("ct chest w/contrast", 36),
        ("ct angiogram of the chest", 35),
        ("ct chest after iv contrast", 32),
        ("ct of the chest without iv contrast", 31),
        ("ct chest findings", 28),
        ("ct angio chest", 28),
        ("ct chest without and with intravenous contrast", 25),
        ("ct of the chest w/iv contrast", 24),
        ("chest findings", 24),
        ("ct chest w/iv contrast", 23),
        ("chest ct without and with iv contrast", 23),
        ("ct angiography of the chest", 22),
        ("ct chest with and without iv contrast", 21),
        ("ct scan of the chest with contrast", 20),
        ("contrast-enhanced ct of the chest", 20),
        ("ct of the chest without contrast", 19),
        ("ct chest without and with contrast", 17),
        ("ct chest with contrast and reconstructions", 17),
        ("chest cta with iv contrast", 16),
        ("post-contrast chest ct", 16),
        ("ct chest following iv contrast", 15),
        ("ct of chest", 14),
        ("cta chest without and with iv contrast", 13),
        ("ct of chest with contrast", 11),
        ("non-contrast chest ct", 11),
        ("cta of the chest without and with iv contrast", 10),
        ("cta of the chest with and without contrast", 9),
        ("ct chest post-administration of intravenous contrast", 9),
        ("cta of the chest with iv contrast", 8),
        ("ct of chest with and without iv contrast", 8),
        ("findings for ct of the chest with iv contrast", 8),
        ("ct of the chest w/contrast", 7),
        ("ct chest with oral and iv contrast", 7),
        ("chest ct w/iv contrast", 7),
        ("ct chest without/with contrast", 7),
        ("cta chest with and without contrast", 7),
        ("cta chest with contrast", 6),
        ("findings for ct of the chest with and without iv contrast", 6),
    ],
    "CT_abdomen-pelvis": [
        ("findings", 5036),
        ("ct of the pelvis with iv contrast", 4252),
        ("ct of the abdomen with iv contrast", 3877),
        ("abdomen", 2750),
        ("ct pelvis", 1703),
        ("pelvis", 1487),
        ("ct of the abdomen without iv contrast", 1065),
        ("ct of the pelvis", 969),
        ("ct pelvis with contrast", 900),
        ("ct abdomen with contrast", 889),
        ("ct of the pelvis without contrast", 883),
        ("ct of the pelvis without iv contrast", 871),
        ("ct of the abdomen without contrast", 869),
        ("ct of the abdomen", 785),
        ("ct of the pelvis with contrast", 762),
        ("ct of the abdomen with contrast", 645),
        ("ct abdomen without iv contrast", 537),
        ("ct pelvis without iv contrast", 477),
        ("ct of the pelvis without intravenous contrast", 340),
        ("ct of the abdomen without intravenous contrast", 310),
        ("ct abdomen without intravenous contrast", 145),
        ("ct pelvis without intravenous contrast", 132),
        ("pelvis with contrast", 118),
        ("ct pelvis w/o iv contrast", 108),
        ("ct abdomen w/o contrast", 98),
        ("ct of the pelvis with oral and iv contrast", 97),
        ("abdomen with contrast", 96),
        ("abdomen ct", 89),
        ("ct of the abdomen with oral and iv contrast", 85),
        ("ct of the pelvis with oral contrast only", 77),
        ("ct abdomen w/contrast", 75),
        ("pelvis ct", 74),
        ("ct pelvis w/contrast", 72),
        ("ct of the abdomen with oral contrast only", 70),
        ("ct of the abdomen with and without iv contrast", 63),
        ("ct pelvis after iv contrast", 61),
        ("ct abdomen findings", 58),
        ("ct of the abdomen with intravenous and oral contrast", 54),
        ("ct abdomen after iv contrast", 50),
        ("ct of the pelvis with intravenous and oral contrast", 50),
        ("ct pelvis findings", 48),
        ("ct of pelvis", 44),
        ("ct of the abdomen without and with iv contrast", 43),
        ("ct abdomen with oral, with intravenous contrast", 42),
        ("ct pelvis with oral, with intravenous contrast", 40),
        ("findings for ct of the pelvis", 40),
        ("ct of abdomen without iv contrast", 39),
        ("ct of the pelvis with iv and oral contrast", 39),
        ("ct of the pelvis w/iv contrast", 36),
        ("non-contrast ct abdomen", 36),
        ("ct of pelvis without iv contrast", 34),
        ("ct pelvis w/iv contrast", 32),
        ("ct abdomen without and with iv contrast", 32),
        ("ct abdomen with contrast and reconstructions", 32),
        ("ct of the pelvis without intravenous or oral contrast", 31),
        ("ct abdomen with and without contrast", 30),
        ("ct pelvis with contrast and reconstructions", 30),
        ("pelvis ct without iv contrast", 29),
        ("ct of the abdomen with oral contrast", 29),
        ("ct of the abdomen without intravenous or oral contrast", 29),
        ("ct of the abdomen with and without contrast", 28),
        ("ct of the abdomen without and with intravenous contrast", 28),
        ("non-contrast ct pelvis", 27),
        ("ct of the pelvis with and without iv contrast", 26),
        ("ct of the pelvis with contrast, findings", 26),
        ("ct of the abdomen without and with contrast", 25),
        ("ct of the abdomen with contrast, findings", 25),
        ("ct of the abdomen with iv and oral contrast", 25),
        ("contrast-enhanced ct of the pelvis", 25),
        ("ct of the pelvis w/o iv contrast", 24),
        ("ct of the abdomen with and without intravenous contrast", 24),
        ("contrast-enhanced ct of the abdomen", 24),
        ("ct abdomen with iv and oral contrast", 23),
        ("ct of abdomen", 23),
        ("ct abdomen with oral, without intravenous contrast", 22),
        ("ct pelvis with iv and oral contrast", 22),
        ("ct of the abdomen w/o iv contrast", 21),
        ("ct of the pelvis w/contrast", 21),
        ("ct abdomen with oral contrast", 21),
        ("non-contrast ct of the abdomen", 21),
        ("ct of the abdomen with oral and intravenous contrast", 21),
        ("ct scan of the pelvis with contrast", 20),
        ("ct pelvis with oral contrast", 20),
        ("ct scan of the abdomen with contrast", 19),
        ("ct of the pelvis with oral contrast", 18),
        ("ct scan of pelvis with oral and intravenous contrast", 18),
        ("abdomen findings", 18),
        ("ct pelvis with oral, without intravenous contrast", 17),
        ("ct of pelvis findings", 17),
        ("ct of the abdomen w/contrast", 16),
        ("ct abdomen with and without intravenous contrast", 16),
        ("ct of the abdomen without iv or oral contrast", 16),
        ("ct of the pelvis without and with iv contrast", 15),
        ("ct abdomen with oral contrast only", 15),
        ("ct of the pelvis without and with intravenous contrast", 15),
        ("cta of the abdomen", 15),
        ("findings for ct of the abdomen without iv contrast", 15),
        ("ct of pelvis with contrast", 14),
        ("ct of abdomen with contrast", 14),
        ("ct of the abdomen w/o contrast", 14),
        ("ct pelvis with oral contrast only", 14),
        ("ct of the pelvis without oral or iv contrast", 14),
        ("ct of the pelvis with oral, with intravenous contrast", 13),
        ("ct scan of abdomen with oral and intravenous contrast", 13),
        ("pelvis findings", 13),
        ("ct of the pelvis with and without contrast", 12),
        ("ct abdomen without and with contrast", 12),
        ("ct abdomen with oral and iv contrast", 12),
        ("non-contrast ct of the pelvis", 12),
        ("abdomen ct with contrast", 11),
        ("post-contrast ct pelvis", 11),
        ("abdomen with oral contrast", 11),
        ("ct abdomen with and without iv contrast", 10),
        ("abdomen ct w/o contrast", 10),
        ("ct of the pelvis with oral and intravenous contrast", 10),
        ("ct abdomen before and after iv contrast", 10),
        ("ct of the pelvis with no iv contrast", 10),
        ("post contrast ct pelvis", 10),
        ("ct pelvis with and without intravenous contrast", 10),
        ("ct abdomen without oral, without intravenous contrast", 10),
        ("ct pelvis post-administration of intravenous contrast", 10),
        ("ct pelvis with oral and iv contrast", 9),
        ("ct pelvis without oral, without intravenous contrast", 9),
        ("ct abdomen post-administration of intravenous contrast", 9),
        ("ct abdomen without contrast and reconstructions", 9),
        ("pelvis post contrast", 8),
        ("ct pelvis with and without contrast", 8),
        ("post-contrast ct abdomen", 8),
        ("pelvis ct without intravenous contrast", 8),
        ("ct abdomen without oral, with intravenous contrast", 8),
        ("ct abdomen without iv or oral contrast", 7),
        ("noncontrast ct pelvis", 7),
        ("ct pelvis without and with iv contrast", 7),
        ("post contrast ct abdomen", 7),
        ("ct of the pelvis without iv or oral contrast", 7),
        ("ct of the pelvis with and without intravenous contrast", 7),
        ("findings for ct of the abdomen with contrast", 7),
        ("cta of the pelvis", 7),
        ("ct abdomen with and w/o contrast", 6),
        ("ct of the pelvis without and with contrast", 6),
        ("pelvis ct w/iv contrast", 6),
        ("ct of the abdomen without oral or iv contrast", 6),
        ("ct scan of pelvis", 6),
        ("cta abdomen", 6),
        ("ct of the pelvis with rectal contrast", 5),
        ("abdomen post contrast", 5),
        ("noncontrast ct abdomen", 5),
        ("pelvis with oral contrast", 5),
        ("abdomen ct with and without contrast", 5),
        ("ct of the pelvis with oral, without intravenous contrast", 5),
        ("ct pelvis without oral, with intravenous contrast", 5),
        ("ct pelvis without iv or oral contrast", 5),
        ("ct abdomen without and with intravenous contrast", 5),
        ("ct abdomen post administration of intravenous contrast", 5),
        ("ct pelvis post administration of intravenous contrast", 5),
    ],
    "MR_abdomen": [
        ("findings", 951),
        ("mr abdomen without and with contrast", 29),
        ("mr abdomen", 27),
        ("mri abdomen", 22),
        ("mri abdomen with contrast", 18),
        ("mri of the abdomen", 12),
        ("abdomen", 12),
        ("mri of the abdomen without and with contrast", 12),
        ("mri of the abdomen with and without intravenous contrast", 10),
        ("mr of the abdomen with iv gadolinium", 10),
        ("mr abdomen with and without contrast/mrcp", 8),
        ("mra abdomen", 8),
        ("mr abdomen with contrast", 7),
        ("mr abdomen with and without contrast", 7),
        ("mr of the abdomen", 7),
        ("mri of the abdomen with contrast", 6),
        ("mri abdomen without contrast", 5),
    ],
    "MR_pelvis": [
        ("findings", 252),
        ("mr pelvis", 12),
        ("mri pelvis", 7),
        ("mri pelvis with contrast", 4),
    ],
    "MR_spine": [
        ("findings", 3384),
        ("thoracic spine", 199),
        ("cervical spine", 170),
        ("lumbar spine", 161),
        ("mri of the cervical spine", 91),
        ("mri of the lumbar spine", 80),
        ("mri of the thoracic spine", 58),
        ("mr cervical spine", 44),
        ("mr thoracic spine", 40),
        ("mr lumbar spine", 36),
        ("thoracic spine mri", 36),
        ("cervical spine mri", 35),
        ("lumbar spine mri", 32),
        ("mr of the cervical spine", 25),
        ("mri cervical spine", 19),
        ("mr of the thoracic spine", 19),
        ("mri lumbar spine", 14),
        ("mr of the lumbar spine", 14),
        ("mri thoracic spine", 14),
        ("mri of the cervical spine without contrast", 12),
        ("cervical", 10),
        ("c-spine", 8),
        ("mri of the lumbar spine without and with contrast", 7),
        ("mri of the lumbar spine with gadolinium", 6),
        ("t-spine", 5),
        ("l-spine", 5),
        ("mri of the cervical spine without and with contrast", 5),
        ("mri of the thoracic spine without gadolinium", 5),
        ("mri of the lumbar spine", 5),
        ("mri of the cervical spine", 4),
        ("c-spine mri without contrast", 4),
        ("mri of the lumbar spine without contrast", 4),
        ("mr c and t-spine", 3),
        ("mr cervical spine without contrast", 3),
        ("cervical mri", 3),
        ("mr lumbar spine without contrast", 3),
        ("mri of the lumbar spine without gadolinium", 3),
        ("cervical spine mri without intravenous contrast", 3),
        ("mr c spine", 2),
        ("lumbosacral spine", 2),
        ("mri of the thoracic and lumbar spine", 2),
        ("mri of the cervical spine with contrast", 2),
        ("mri of the cervical spine with and without contrast", 2),
        ("mr of the lumbar spine with and without contrast", 2),
        ("mri of the cervical spine with gadolinium", 2),
        ("c-spine mr", 2),
        ("mri of the lumbar spine without and with gadolinium", 2),
        ("mri of the thoracic spine without gadolinium", 2),
        ("mri l-spine without contrast", 2),
        ("mr cervical spine with and without iv contrast", 2),
        ("cervical spine findings", 2),
        ("lumbar spine findings", 2),
        ("mr cervical spine without iv contrast", 2),
        ("mr thoracic spine with and without iv contrast", 2),
        ("mri l-spine with and without iv contrast", 2),
    ],
    "MR_head": [
        ("findings", 9435),
        ("mra of the head", 479),
        ("mra head", 256),
        ("mri head", 85),
        ("head mri", 82),
        ("head mra", 70),
        ("mri of the head", 38),
        ("mrv of the head", 38),
        ("mr head", 34),
        ("mr head w/o contrast", 24),
        ("mrv head", 20),
        ("mr angiogram of the head", 18),
        ("mri of the brain and mra of the head", 13),
        ("mr head without contrast", 10),
        ("mr of the head", 10),
        ("mr head without and with contrast", 9),
        ("mri of the head without contrast", 9),
        ("head mrv", 9),
        ("mr head without and with iv contrast", 9),
        ("mr head with and without contrast", 6),
        ("mra of head", 6),
        ("mr of the head without contrast", 5),
        ("mri head with contrast", 5),
    ],
    "MR_neck": [
        ("findings", 275),
        ("mra of the neck", 20),
        ("mra neck", 12),
        ("mr angiography of the neck arterial vasculature", 3),
        ("mri of the neck with gadolinium", 3),
        ("mri of the neck", 2),
        ("mra neck with and without contrast", 2),
        ("mr angiogram of the neck", 2),
        ("mri/mra of the neck", 1),
        ("mra neck findings", 1),
        ("mr of the neck", 1),
        ("mr neck without and with contrast", 1),
        ("mri and mra of the neck and aortic arch", 1),
        ("mr of neck", 1),
        ("mr of the neck without and with contrast", 1),
        ("mri scan of the neck with gadolinium enhancement", 1),
        ("mra of the neck w/gadolinium", 1),
        ("mri scan of the soft tissues of the neck", 1),
        ("mri of neck and skull base", 1),
        ("mri of the neck without and with gadolinium", 1),
        ("mr neck", 1),
        ("mr angiogram of the vessels of the neck", 1),
        ("mr neck without gadolinium", 1),
        ("neck mra", 1),
        ("mri neck, before and after iv contrast", 1),
        ("mr neck with iv contrast", 1),
    ],
}


section_map_rev = {}
for k in section_map:
    for v in section_map[k]:
        section_map_rev[v] = k
