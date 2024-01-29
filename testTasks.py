from multimedeval import Params, MultiMedEval

from tensorboardX import SummaryWriter

def batcher(prompts):
    print(prompts[0])
    return ["entailment" for _ in range(len(prompts))]

writer = None # SummaryWriter("testTensorboard")

engine = MultiMedEval(
    params=Params(batch_size=64, fewshot=True, num_workers=8, tensorBoardWriter=writer), batcher=batcher, generateVisualization=True
)

# engine.eval(["MNIST-Oct", "MedQA"])
