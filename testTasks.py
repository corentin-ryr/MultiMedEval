from multimedbench import Params, MMB


def batcher(prompts):
    return ["This CT scan shows the lungs" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=64, fewshot=True), batcher=batcher, generateVisualization=False)

engine.eval(["VQA-RAD", "Path-VQA", "SLAKE"])
