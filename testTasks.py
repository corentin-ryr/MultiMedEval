from multimedbench import Params, MMB


def batcher(prompts):
    return ["yes" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=64, fewshot=False), batcher=batcher, generateVisualization=False)

engine.eval(["VQA-RAD", "Path-VQA", "SLAKE"])
