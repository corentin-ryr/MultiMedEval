from multimedbench.qa import MedQA, PubMedQA, MedMCQA

medqa = MedMCQA()

medqa.isValid("The answer is C: Constriction of afferent arteriole decreases the blood flow to the glomeruli. Glomerular capillaries are specialized blood vessels that filter waste products and excess fluids from the blood. The oncotic pressure of the fluid leaving the capillaries is less than that of fluid entering it, which helps to drive filtration. The glucose concentration in the capillaries is different from that in the gl", medqa.dataset[1])