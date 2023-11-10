from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "medalpaca/medalpaca-7b", device_map = "cuda:0", truncation_side = "left"
)
model = ORTModelForCausalLM.from_pretrained("/home/croyer/data/medalpaca7B-optimized")

optimum_qa = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda:0"
)

prediction = optimum_qa("What's my name?")

print(prediction)
