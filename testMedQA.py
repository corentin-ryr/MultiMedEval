from multimedbench.utils import Params
from multimedbench.engine import MMB


params = Params(True, 42, 64)


def batcher(prompts):
    return ["answer b is correct" for _ in range(len(prompts))]


engine = MMB(params=params, batcher=batcher)

print(engine.eval("PubMedQA")[0])

test = (
    [
        {
            "role": "user",
            "content": "Answer the following question with yes, no or maybe.Dyschesia can be provoked by inappropriate defecation movements. The aim of this prospective study was to demonstrate dysfunction of the anal sphincter and/or the musculus (m.) puborectalis in patients with dyschesia using anorectal endosonography. Twenty consecutive patients with a medical history of dyschesia and a control group of 20 healthy subjects underwent linear anorectal endosonography (Toshiba models IUV 5060 and PVL-625 RT). In both groups, the dimensions of the anal sphincter and the m. puborectalis were measured at rest, and during voluntary squeezing and straining. Statistical analysis was performed within and between the two groups. The anal sphincter became paradoxically shorter and/or thicker during straining (versus the resting state) in 85% of patients but in only 35% of control subjects. Changes in sphincter length were statistically significantly different (p<0.01, chi(2) test) in patients compared with control subjects. The m. puborectalis became paradoxically shorter and/or thicker during straining in 80% of patients but in only 30% of controls. Both the changes in length and thickness of the m. puborectalis were significantly different (p<0.01, chi(2) test) in patients versus control subjects.\nQuestion: Is anorectal endosonography valuable in dyschesia?\n",
        }
    ],
    [],
)
