from typing import Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralModel,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)
from multimedbench.engine import MMB
from multimedbench.utils import Params
from transformers import pipeline
import time
import os


class batcherMistral:
    def __init__(self, device=None) -> None:
        self.device = device

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        nf4_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model: MistralModel = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            quantization_config=nf4_config,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.pl = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     device_map=self.device,
        # )

    def __call__(self, prompts):
        answers = []

        model_inputs = [
            self.tokenizer.apply_chat_template(messages[0], return_tensors="pt", tokenize=False) for messages in prompts
        ]
        model_inputs = self.tokenizer(
            model_inputs, padding="max_length", truncation=True, max_length=1024, return_tensors="pt"
        )
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=self.tokenizer.pad_token_id
        )

        # Remove the first 1024 tokens (prompt)
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]

        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return answers


class batcherLlama:
    def __init__(self, device=None) -> None:
        self.device = "auto" if device is None else device
        self.loadModelAndTokenizer()
        self.model.eval()

    def loadModelAndTokenizer(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        nf4_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            device_map=self.device,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )
        self.generation_config = None

        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", padding_side="left", truncation_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompts):
        model_inputs = [
            self.tokenizer.apply_chat_template(messages[0], return_tensors="pt", tokenize=False) for messages in prompts
        ]

        print(model_inputs)

        with torch.no_grad():
            tokens = self.tokenizer(
                model_inputs,
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            tokens = {k: v.to("cuda") for k, v in tokens.items()}

            startTime = time.time()
            answerTokens = self.model.generate(**tokens, max_new_tokens=100, generation_config=self.generation_config)
            execTime = time.time() - startTime

            answerTokens = answerTokens[:, tokens["input_ids"].shape[1] :]
            answers = self.tokenizer.batch_decode(answerTokens, skip_special_tokens=True)

        numberTokens = torch.count_nonzero(answerTokens > 2)
        print(
            f"Model inference: {execTime:.2f}s to generate {numberTokens} tokens ({numberTokens / execTime:.2f} t/s.)"
        )

        # print(answers)

        # raise Exception


        return answers


class batcherMedAlpaca(batcherLlama):
    def loadModelAndTokenizer(self) -> None:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        nf4_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = LlamaForCausalLM.from_pretrained(
            "medalpaca/medalpaca-7b",
            device_map=self.device,
            # quantization_config=nf4_config,
            torch_dtype=torch.float16,
        )

        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
            "medalpaca/medalpaca-7b",
            truncation_side="left",
            padding_side="left",
        )
        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        self.tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n' + content.strip() + '\n\n' + '### Answer:\n' }}{% elif message['role'] == 'assistant' %}{{ content.strip() + eos_token }}{% endif %}{% endfor %}"""


class batcherPMCLlama(batcherLlama):
    def loadModelAndTokenizer(self):
        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained("axiong/PMC_LLaMA_13B", padding_side="left")

        nf4_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = LlamaForCausalLM.from_pretrained(
            "axiong/PMC_LLaMA_13B", torch_dtype=torch.float16, device_map="cuda"
        )  # , quantization_config=nf4_config)
        # self.generation_config = None  # GenerationConfig(eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eos_token_id = self.tokenizer.eos_token_id
        self.model.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Input:\n'  + message['content'] + "\n### Answer:" }}{% elif message['role'] == 'system' %}{{ '###Instruction:\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""

    def __call__(self, prompts):
        instruction = "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly."

        prompts = [[{"role": "system", "content": instruction}] + prompt[0] for prompt in prompts]

        input_str = [
            self.tokenizer.apply_chat_template(prompt, return_tensors="pt", tokenize=False) for prompt in prompts
        ]

        model_inputs = self.tokenizer(
            input_str,
            return_tensors="pt",
            padding=True,
        )

        # Move model_inputs to cuda
        model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        topk_output = self.model.generate(
            **model_inputs, max_new_tokens=1000, top_k=50, pad_token_id=self.tokenizer.pad_token_id
        )

        # Remove the first 1024 tokens (prompt)
        topk_output = topk_output[:, model_inputs["input_ids"].shape[1] :]

        output_str = self.tokenizer.batch_decode(topk_output, skip_special_tokens=True)

        return output_str


if __name__ == "__main__":
    params = Params(True, seed=42, batch_size=8, run_name="benchmarkMedAlpaca")

    device = "cuda:0"
    batcher = batcherMedAlpaca(device=device)

    mmb = MMB(params, batcher, fewshot=False)

    results = mmb.eval(["PubMedQA", "MedMCQA", "MedQA"])
    print(f"Everything is done, see the {params.run_name} folder for detailed results.")
