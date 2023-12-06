from multimedbench.mimic import MIMIC_CXR_reportgen
from multimedbench.utils import Params


import json
from scorers.RadGraph.RadGraph import RadGraph
refs = ["no evidence of consolidation to suggest pneumonia is seen . there is some retrocardiac atelectasis . a small left pleural effusion may be present . no pneumothorax is seen . no pulmonary edema . a right granuloma is unchanged . the heart is mildly enlarged , unchanged . there is tortuosity of the aorta ."]
hyps = ["heart size is enlarged . mediastinal silhouette and hilar contours are unchanged . the lung volumes are low . there is mild opacity of the left lung . small right pleural effusion is present compared to the prior exam . there is mild atelectasis of the right lung . there is no pneumothorax is demonstrated ."]
mean_reward, _, hypothesis_annotation_lists, reference_annotation_lists = RadGraph(reward_level="partial", cuda=-1)(
    refs=refs, hyps=hyps)

print(json.dumps(reference_annotation_lists, indent=4))
print(json.dumps(hypothesis_annotation_lists, indent=4))
print("reward", mean_reward)


raise Exception


params = Params(True, 42, 64)

def batcher(prompts):
    return ["yes" for _ in range(len(prompts))]

mimic = MIMIC_CXR_reportgen()

print(mimic.run(params, batcher)[0])