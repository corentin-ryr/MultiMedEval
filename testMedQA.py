from multimedbench.qa import MedQA
from multimedbench.utils import Params

vqa = MedQA()

print(vqa.format_question(vqa.dataset[0]))

vqa.run(Params(batch_size=2), batcher=lambda x: x)
