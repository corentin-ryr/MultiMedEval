from multimedbench import Params, MMB


def batcher(prompts):
    return ["entailment" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=64, fewshot=True, num_workers=8), batcher=batcher, generateVisualization=False)

# engine.eval([])
