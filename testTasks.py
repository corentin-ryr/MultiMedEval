from multimedbench import Params, MMB


def batcher(prompts):
    return ["0" for _ in range(len(prompts))]


engine = MMB(params=Params(batch_size=4, fewshot=False), batcher=batcher, generateVisualization=False)

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
    ]
)
