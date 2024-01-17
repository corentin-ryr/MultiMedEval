from multimedbench import Params, MMB

def batcher(prompts):
    return ["" for _ in range(len(prompts))]

engine = MMB(params=Params(batch_size=64, fewshot=True), batcher=batcher, generateVisualization=True)

engine.eval(["MIMIC-CXR-ImageClassification","VinDr-Mammo","Pad-UFES-20","CBIS-DDSM"])
