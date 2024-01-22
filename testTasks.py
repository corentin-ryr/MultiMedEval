from multimedbench import Params, MMB


def batcher(prompts):
    return ["0" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=4, fewshot=True), batcher=batcher, generateVisualization=False)

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
