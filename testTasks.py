from multimedbench import Params, MMB

from nltk.stem import WordNetLemmatizer


def batcher(prompts):
    return ["0" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=64, fewshot=True, num_workers=8), batcher=batcher, generateVisualization=True)

engine.eval(
    [
        "MNIST-Oct",
        "MNIST-Path",
        "MNIST-Blood",
        "MNIST-Breast",
        "MNIST-Derma",
        "MNIST-OrganA",
        "MNIST-Chest",
        "MNIST-OrganC",
        "MNIST-OrganS",
        "MNIST-Pneumonia",
        "MNIST-Retina",
        "MNIST-Tissue",
        "MedQA",
        "PubMedQA",
        "MedMCQA",
        "MIMIC-CXR-ReportGeneration",
        "VQA-RAD",
        "Path-VQA",
        "SLAKE",
        "MIMIC-CXR-ImageClassification",
        "VinDr-Mammo",
        "Pad-UFES-20",
        "CBIS-DDSM-Mass",
        "CBIS-DDSM-Calcification",
        "MIMIC-III",
    ]
)
