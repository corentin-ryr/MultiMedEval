from multimedbench import Params, MMB

def batcher(prompts):
    return ["" for _ in range(len(prompts))]

engine = MMB(params=Params(batch_size=64), batcher=batcher, generateVisualization=True)

raise Exception
engine.eval(["MedQA","PubMedQA","MedMCQA","MIMIC-CXR-ReportGeneration",
             "VQA-RAD","Path-VQA","SLAKE","MIMIC-CXR-ImageClassification",
             "VinDr-Mammo","Pad-UFES-20","CBIS-DDSM","MIMIC-III"])
