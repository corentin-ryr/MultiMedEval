from multimedeval import Params, MultiMedEval


def batcher(prompts):
    print(prompts[0])
    return ["entailment" for _ in range(len(prompts))]


engine = MultiMedEval(
    params=Params(batch_size=64, fewshot=True, num_workers=8), batcher=batcher, generateVisualization=False
)

engine.eval(["MNIST-Oct"])
