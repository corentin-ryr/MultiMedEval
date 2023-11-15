
from multimedbench.vqa import VQA_RAD

vqa = VQA_RAD()

print(vqa.format_question(vqa.dataset[4]))   

print(vqa.isValid("it is not here", vqa.dataset[4]))

    