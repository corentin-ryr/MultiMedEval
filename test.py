import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralModel, BitsAndBytesConfig
import os

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = "cuda" # the device to load the model onto
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

model:MistralModel = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=nf4_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "Hello, how are you?"

model_inputs = tokenizer([prompt], return_tensors="pt")
print(model_inputs)


# model.to(device)


generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])