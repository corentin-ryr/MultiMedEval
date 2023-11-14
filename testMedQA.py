from multimedbench.medqa import MedQA, PubMedQA, MedMCQA

medqa = MedMCQA(data_folder="data")

print(medqa.prompt)

test = [
    {
        "role": "user",
        "content": "Question: In a survey, many children are examined and were found to have urogenital abnormalities. Which congenital anomaly is associated with increased risk of bladder carcinoma?\nOptions:\nA: Medullary sponge kidney\nB: Bladder extrophy\nC: Unilateral renal agenesis\nD: Double ureter",
    },
    {"role": "assistant", "content": "The answer is B: Bladder extrophy."},
    {
        "role": "user",
        "content": "Question: Under normal condition amount of fluoride secreted from the gland in the mouth:\nOptions:\nA: 0.007-0.05ppm\nB: 0.005-0.007ppm\nC: 0.07-0.05ppm\nD: None.",
    },
    {"role": "assistant", "content": "The answer is A: 0.007-0.05ppm."},
    {
        "role": "user",
        "content": "Question: Damage to pneumotaxic center produces ?\nOptions:\nA: Deep and fast respiration\nB: Deep and slow respiration\nC: Shallow and fast respiration\nD: Shallow and slow respiration",
    },
    {"role": "assistant", "content": "The answer is B: Deep and slow respiration."},
]
