import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
from optimum.onnxruntime import ORTOptimizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import OptimizationConfig


model_id = "medalpaca/medalpaca-7b"
task = "text-generation"


# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     device_map="cuda:0",
#     truncation_side="left",
#     padding_side="left",
# )
# # quantization_config = GPTQConfig(bits=4, dataset="wikitext2", tokenizer=tokenizer)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# print("Config ready")

# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map="cuda:0", quantization_config=quantization_config
# )
# model.save_pretrained(Path("/home/croyer/data/medalpaca7B-quantized"))
# print("Model saved")

print("Optimization.")
model = ORTModelForCausalLM.from_pretrained(Path("/home/croyer/data/medalpaca7B-quantized"), use_safetensors=True)
# create ORTOptimizer and define optimization configuration
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=99
)  # enable all optimizations

# apply the optimization configuration to the model
path = optimizer.optimize(
    optimization_config=optimization_config,
    save_dir=Path("/home/croyer/data/medalpaca7B-quantized-optimized"),
)
