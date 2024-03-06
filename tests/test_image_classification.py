from multimedeval import MultiMedEval, SetupParams, EvalParams
import json


def batcher(prompts):
    return ["Dummy answer" for _ in range(len(prompts))]


def test_image_classification():
    engine = MultiMedEval()
    config = json.load(open("MedMD_config.json"))
    engine.setup(SetupParams(VinDr_Mammo_dir=config["VinDr_Mammo_dir"]))
    
    
    engine.eval(["VinDr Mammo"], batcher, EvalParams())