from multimedbench.vqa import VQA_RAD, Path_VQA, SLAKE
from multimedbench.utils import Params



if __name__ == "__main__":
    params = Params(True, 42, 64)

    def batcher(prompts):
        return ["yes" for _ in range(len(prompts))]

    vqa_rad = VQA_RAD()
    print(vqa_rad.run(params, batcher)[0])

    vqa_rad = Path_VQA()
    print(vqa_rad.run(params, batcher)[0])

    vqa_rad = SLAKE()
    print(vqa_rad.run(params, batcher)[0])

